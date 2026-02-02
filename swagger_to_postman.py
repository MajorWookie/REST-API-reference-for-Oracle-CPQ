#!/usr/bin/env python3
"""
Convert a Swagger 2.0 JSON file to a Postman Collection v2.1.0 JSON file.

Designed for Oracle CPQ REST APIs. Produces a collection that uses environment
variables for site URL, REST version, and credentials, and organizes endpoints
into nested folders by Swagger tags and URL path hierarchy.

Usage:
    python swagger_to_postman.py CPQ/cxcpqswagger.json
    python swagger_to_postman.py CPQ/cxcpqswagger.json -o output.postman_collection.json
    python swagger_to_postman.py CPQ/cxcpqswagger.json -n "My Collection" --rest-version-value "rest/v19"
"""

import argparse
import copy
import json
import os
import re
import sys
import uuid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSTMAN_SCHEMA = "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"

# Swagger path parameters that map to Postman environment variables rather
# than per-request path variables. When these names appear as {paramName} in
# a Swagger path, they are emitted as {{ParamName}} in the Postman URL instead
# of :paramName.
CPQ_ENV_VARIABLE_PARAMS = {
    "Stage": "{{Stage}}",
    "ProcessVarName": "{{ProcessVarName}}",
    "MainDocVarName": "{{MainDocVarName}}",
    "SubDocVarName": "{{SubDocVarName}}",
    "ProdFamVarName": "{{ProdFamVarName}}",
    "ProdLineVarName": "{{ProdLineVarName}}",
    "ModelVarName": "{{ModelVarName}}",
    "ArraySetVarName": "{{ArraySetVarName}}",
}

# HTTP status text lookup
HTTP_STATUS_TEXT = {
    200: "OK",
    201: "Created",
    202: "Accepted",
    204: "No Content",
    301: "Moved Permanently",
    304: "Not Modified",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    409: "Conflict",
    415: "Unsupported Media Type",
    500: "Internal Server Error",
}

# Placeholder values by Swagger type
TYPE_PLACEHOLDERS = {
    "string": "<string>",
    "integer": "<integer>",
    "number": "<number>",
    "boolean": "<boolean>",
    "file": "",
}

MAX_SCHEMA_DEPTH = 5


# ---------------------------------------------------------------------------
# Name Formatting (consolidated from postman_collection_formatter.py)
# ---------------------------------------------------------------------------


def format_name(name):
    """Convert camelCase or snake_case to Title Case with spaces.

    Examples:
        productFamilies     -> Product Families
        bomItemSetups       -> Bom Item Setups
        allProductFamilySetups -> All Product Family Setups
        user_navigation_links  -> User Navigation Links

    Names that are path parameter placeholders ({paramName}, :paramName) or
    environment variable references ({{VarName}}) are left unchanged.
    Segments containing mixed text and {{vars}} only format the text portions.
    """
    if not isinstance(name, str):
        return name
    # Skip names that are entirely a path param or env var reference
    if name.startswith(":") or re.match(r"^\{[^}]+\}$", name) or re.match(r"^\{\{[^}]+\}\}$", name):
        return name

    # If the name contains {{...}} or {...} references, only format the
    # non-variable portions to preserve variable names exactly.
    if "{{" in name or ("{" in name and not name.startswith("{")):
        # Split on variable references, preserving them
        parts = re.split(r"(\{\{[^}]+\}\}|\{[^}]+\})", name)
        result = []
        for part in parts:
            if part.startswith("{{") or part.startswith("{"):
                result.append(part)
            else:
                # Format the non-variable text
                formatted = re.sub(r"([a-z])([A-Z])", r"\1 \2", part)
                formatted = formatted.replace("_", " ")
                result.append(formatted.title())
        return "".join(result)

    # Standard formatting: split camelCase, replace underscores, title case
    formatted = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    formatted = formatted.replace("_", " ")
    return formatted.title()


# ---------------------------------------------------------------------------
# Schema / Reference Resolution
# ---------------------------------------------------------------------------


def resolve_ref(ref_string, definitions):
    """Follow a $ref like '#/definitions/Foo' and return the definition dict."""
    if not ref_string or not ref_string.startswith("#/definitions/"):
        return {}
    name = ref_string[len("#/definitions/"):]
    return definitions.get(name, {}), name


def resolve_schema(schema, definitions, depth=0, seen=None):
    """Recursively resolve a Swagger schema, following $ref and merging allOf."""
    if seen is None:
        seen = set()
    if depth > MAX_SCHEMA_DEPTH:
        return schema

    if "$ref" in schema:
        resolved, name = resolve_ref(schema["$ref"], definitions)
        if name in seen:
            return {"type": "object"}
        seen = seen | {name}
        return resolve_schema(resolved, definitions, depth + 1, seen)

    if "allOf" in schema:
        merged = {"type": "object", "properties": {}}
        for sub in schema["allOf"]:
            resolved = resolve_schema(sub, definitions, depth + 1, seen)
            if "properties" in resolved:
                merged["properties"].update(resolved["properties"])
        return merged

    return schema


def schema_to_example(schema, definitions, depth=0, seen=None):
    """Convert a Swagger schema to an example value with type placeholders."""
    if seen is None:
        seen = set()
    if depth > MAX_SCHEMA_DEPTH:
        return {}

    if "$ref" in schema:
        resolved, name = resolve_ref(schema["$ref"], definitions)
        if name in seen:
            return {}
        seen = seen | {name}
        return schema_to_example(resolved, definitions, depth + 1, seen)

    if "allOf" in schema:
        merged = {}
        for sub in schema["allOf"]:
            result = schema_to_example(sub, definitions, depth + 1, seen)
            if isinstance(result, dict):
                merged.update(result)
        return merged

    schema_type = schema.get("type", "object")

    if schema_type == "array":
        items_schema = schema.get("items", {})
        item_example = schema_to_example(items_schema, definitions, depth + 1, seen)
        return [item_example]

    if schema_type == "object" or "properties" in schema:
        obj = {}
        for prop_name, prop_schema in schema.get("properties", {}).items():
            obj[prop_name] = schema_to_example(prop_schema, definitions, depth + 1, seen)
        return obj

    # Primitive types
    if "enum" in schema:
        return schema["enum"][0] if schema["enum"] else TYPE_PLACEHOLDERS.get(schema_type, "<string>")

    return TYPE_PLACEHOLDERS.get(schema_type, "<string>")


# ---------------------------------------------------------------------------
# Path and Variable Processing
# ---------------------------------------------------------------------------


def extract_version_prefix(path):
    """
    Extract the REST version prefix from a path like '/rest/v19/accounts/{id}'.

    Returns (version_path, remaining_path) where version_path is e.g. 'rest/v19'
    and remaining_path is e.g. 'accounts/{id}'.
    If no version prefix is found, returns (None, full_path_without_leading_slash).
    """
    match = re.match(r"^/?(rest/v\d+)/(.*)", path)
    if match:
        return match.group(1), match.group(2)
    # Fallback: strip leading slash
    return None, path.lstrip("/")


def path_to_postman_segments(remaining_path):
    """
    Convert a path string to Postman URL segments.

    Replaces {paramName} with :paramName unless it's a known CPQ env variable,
    in which case it becomes {{ParamName}}.
    """
    if not remaining_path:
        return []

    segments = []
    for seg in remaining_path.split("/"):
        if not seg:
            continue
        # Handle segments that contain one or more {param} placeholders
        # e.g. "commerce{Stage}{ProcessVarName}{MainDocVarName}"
        # or a simple "{id}"
        parts = re.split(r"(\{[^}]+\})", seg)
        built = ""
        for part in parts:
            param_match = re.match(r"^\{(.+)\}$", part)
            if param_match:
                param_name = param_match.group(1)
                if param_name in CPQ_ENV_VARIABLE_PARAMS:
                    built += CPQ_ENV_VARIABLE_PARAMS[param_name]
                else:
                    # If there's text before this param, flush it as its own segment
                    if built:
                        segments.append(built)
                        built = ""
                    segments.append(":" + param_name)
            else:
                built += part

        if built:
            segments.append(built)

    return segments


def build_postman_url(path, path_params, query_params, definitions):
    """Build a complete Postman URL object from a Swagger path and parameters."""
    version_path, remaining = extract_version_prefix(path)

    segments = path_to_postman_segments(remaining)

    # Build path array: {{RestVersion}} first (if version found), then segments
    path_array = []
    if version_path:
        path_array.append("{{RestVersion}}")
    path_array.extend(segments)

    # Build raw URL string
    raw = "{{CPQ Site URL}}/" + "/".join(path_array)

    # Build path variables (exclude CPQ env variable params)
    variables = []
    for param in path_params:
        name = param.get("name", "")
        if name in CPQ_ENV_VARIABLE_PARAMS:
            continue
        desc = param.get("description", "")
        if param.get("required", False) and not desc.startswith("(Required)"):
            desc = "(Required) " + desc
        variables.append({
            "key": name,
            "value": TYPE_PLACEHOLDERS.get(param.get("type", "string"), "<string>"),
            "description": desc,
        })

    # Build query parameters
    query = []
    for param in query_params:
        entry = {
            "key": param.get("name", ""),
            "value": TYPE_PLACEHOLDERS.get(param.get("type", "string"), "<string>"),
            "description": param.get("description", ""),
        }
        if not param.get("required", False):
            entry["disabled"] = True
        query.append(entry)

    url_obj = {
        "raw": raw,
        "host": ["{{CPQ Site URL}}"],
        "path": path_array,
    }
    if variables:
        url_obj["variable"] = variables
    if query:
        # Append query string to raw URL
        query_parts = [q["key"] + "=" + q["value"] for q in query if not q.get("disabled")]
        if query_parts:
            url_obj["raw"] = raw + "?" + "&".join(query_parts)
        url_obj["query"] = query

    return url_obj


# ---------------------------------------------------------------------------
# Request Building
# ---------------------------------------------------------------------------


def build_request_headers(method, operation, has_body, is_form_data):
    """Build Postman request headers from a Swagger operation."""
    headers = []

    # Accept header
    produces = operation.get("produces", ["application/json"])
    accept = produces[0] if produces else "application/json"
    headers.append({"key": "Accept", "value": accept})

    # Content-Type header for methods with bodies
    if has_body and method in ("POST", "PUT", "PATCH"):
        if is_form_data:
            headers.append({"key": "Content-Type", "value": "multipart/form-data"})
        else:
            consumes = operation.get("consumes", ["application/json"])
            content_type = consumes[0] if consumes else "application/json"
            headers.append({"key": "Content-Type", "value": content_type})

    # Header parameters from Swagger
    for param in operation.get("parameters", []):
        if param.get("in") == "header":
            headers.append({
                "key": param.get("name", ""),
                "value": TYPE_PLACEHOLDERS.get(param.get("type", "string"), "<string>"),
                "description": param.get("description", ""),
            })

    return headers


def build_request_body(operation, definitions):
    """Build a Postman request body from Swagger parameters."""
    params = operation.get("parameters", [])

    # Check for formData parameters
    form_params = [p for p in params if p.get("in") == "formData"]
    if form_params:
        formdata = []
        for param in form_params:
            if param.get("type") == "file":
                formdata.append({
                    "key": param.get("name", "file"),
                    "type": "file",
                    "src": "",
                    "description": param.get("description", ""),
                })
            else:
                formdata.append({
                    "key": param.get("name", ""),
                    "value": TYPE_PLACEHOLDERS.get(param.get("type", "string"), "<string>"),
                    "type": "text",
                    "description": param.get("description", ""),
                })
        return {"mode": "formdata", "formdata": formdata}, True

    # Check for body parameter
    body_params = [p for p in params if p.get("in") == "body"]
    if body_params:
        body_param = body_params[0]
        schema = body_param.get("schema", {})
        example = schema_to_example(schema, definitions)
        return {
            "mode": "raw",
            "raw": json.dumps(example, indent=2),
            "options": {"raw": {"language": "json"}},
        }, False

    return None, False


def build_response_examples(method, path, operation, request_obj, definitions):
    """Build Postman response examples from Swagger response definitions."""
    responses = operation.get("responses", {})
    examples = []

    for status_code_str, response_def in responses.items():
        # Determine numeric status code
        if status_code_str == "default":
            code = 200
        else:
            try:
                code = int(status_code_str)
            except ValueError:
                code = 200

        status_text = HTTP_STATUS_TEXT.get(code, "Unknown")
        description = response_def.get("description", status_text)

        # Build response body from schema
        body = ""
        schema = response_def.get("schema")
        if schema:
            example = schema_to_example(schema, definitions)
            body = json.dumps(example, indent=2)

        # Build originalRequest (simplified copy)
        orig_request = {
            "method": method.upper(),
            "header": copy.deepcopy(request_obj.get("header", [])),
            "url": copy.deepcopy(request_obj.get("url", {})),
        }
        if "body" in request_obj:
            orig_request["body"] = copy.deepcopy(request_obj["body"])

        example_obj = {
            "name": description if description else status_text,
            "originalRequest": orig_request,
            "status": status_text,
            "code": code,
            "_postman_previewlanguage": "json",
            "header": [{"key": "Content-Type", "value": "application/json"}],
            "cookie": [],
            "body": body,
        }
        examples.append(example_obj)

    return examples


def build_request_item(method, path, operation, definitions, global_consumes=None, global_produces=None):
    """Assemble a complete Postman request item from a Swagger operation."""
    # Merge global consumes/produces if operation doesn't specify its own
    if global_consumes and "consumes" not in operation:
        operation = dict(operation)
        operation["consumes"] = global_consumes
    if global_produces and "produces" not in operation:
        operation = dict(operation)
        operation["produces"] = global_produces

    params = operation.get("parameters", [])
    path_params = [p for p in params if p.get("in") == "path"]
    query_params = [p for p in params if p.get("in") == "query"]

    # Build URL
    url_obj = build_postman_url(path, path_params, query_params, definitions)

    # Build body
    body, is_form_data = build_request_body(operation, definitions)
    has_body = body is not None

    # Build headers
    headers = build_request_headers(method.upper(), operation, has_body, is_form_data)

    # Name from summary, operationId, or path
    name = operation.get("summary", "")
    if not name:
        name = operation.get("operationId", "")
    if not name:
        name = f"{method.upper()} {path}"

    request_obj = {
        "method": method.upper(),
        "header": headers,
        "url": url_obj,
    }
    if body:
        request_obj["body"] = body

    desc = operation.get("description", "")
    if desc:
        request_obj["description"] = desc

    # Build response examples
    response_examples = build_response_examples(
        method, path, operation, request_obj, definitions
    )

    item = {
        "name": name,
        "request": request_obj,
        "response": response_examples,
    }

    return item


# ---------------------------------------------------------------------------
# Folder Organization
# ---------------------------------------------------------------------------


def group_operations_by_tag(swagger):
    """
    Group all Swagger operations by their first tag.

    Returns dict: { "tag_name": [(method, path, operation), ...] }
    """
    paths = swagger.get("paths", {})
    groups = {}

    for path, path_item in paths.items():
        # Collect path-level parameters (shared across all methods)
        path_level_params = path_item.get("parameters", [])

        for method in ("get", "post", "put", "patch", "delete", "options", "head"):
            if method not in path_item:
                continue
            operation = path_item[method]

            # Merge path-level parameters with operation-level parameters
            op_params = operation.get("parameters", [])
            op_param_names = {(p.get("name"), p.get("in")) for p in op_params}
            merged_params = list(op_params)
            for pp in path_level_params:
                if (pp.get("name"), pp.get("in")) not in op_param_names:
                    merged_params.append(pp)
            operation = dict(operation)
            operation["parameters"] = merged_params

            tags = operation.get("tags", ["Other"])
            tag = tags[0] if tags else "Other"
            groups.setdefault(tag, []).append((method, path, operation))

    return groups


def insert_into_tree(tree, path_segments, item):
    """
    Insert a request item into a nested folder tree dict.

    tree is a dict: { "_items": [...], "segment_name": sub_tree_dict, ... }
    path_segments is a list of folder names to nest into.
    item is the Postman request item dict.
    """
    node = tree
    for seg in path_segments:
        if seg not in node:
            node[seg] = {"_items": []}
        node = node[seg]
    node["_items"].append(item)


def tree_to_postman_items(tree, depth=0):
    """
    Convert a nested tree dict into a Postman item array (folders + requests).

    Folders (sub-trees) come first, then leaf request items.
    """
    items = []
    # Sub-folders
    for key, subtree in sorted(tree.items()):
        if key == "_items":
            continue
        folder = {
            "name": key,
            "item": tree_to_postman_items(subtree, depth + 1),
        }
        items.append(folder)
    # Leaf items
    items.extend(tree.get("_items", []))
    return items


def build_folder_for_tag(tag_name, operations, definitions, tag_descriptions,
                         global_consumes=None, global_produces=None):
    """
    Build a Postman folder (or nested folders) for a single Swagger tag.

    The tag name may contain '/' to indicate hierarchy (e.g. 'Commerce/Actions').
    Within the deepest tag folder, endpoints are nested by their URL path segments.
    """
    # Build request items and determine their path-based nesting
    tree = {"_items": []}

    for method, path, operation in operations:
        item = build_request_item(method, path, operation, definitions,
                                  global_consumes, global_produces)

        # Determine sub-folder path from URL segments
        _, remaining = extract_version_prefix(path)
        raw_segments = remaining.split("/")
        # Clean segments: remove empty, remove pure param segments for folder naming
        folder_segments = []
        for seg in raw_segments:
            if not seg:
                continue
            # Keep segments as folder names (including {param} segments)
            folder_segments.append(seg)

        # The last segment(s) that correspond to the request itself shouldn't be folders.
        # Strategy: all segments except the trailing resource/param combo become folders.
        # If path ends with a param like {id}, the item goes into parent folder.
        # If path ends with a resource name, that resource is also a folder level.
        # We put the request as a leaf inside the full path hierarchy.
        insert_into_tree(tree, folder_segments, item)

    # Convert tree to Postman items
    postman_items = tree_to_postman_items(tree)

    # Collapse single-child folder chains for cleaner structure
    postman_items = collapse_single_child_folders(postman_items)

    # Build the tag folder hierarchy
    tag_parts = tag_name.split("/")
    tag_desc = tag_descriptions.get(tag_name, "")

    # Wrap in tag folder hierarchy (innermost first)
    result = postman_items
    for i in range(len(tag_parts) - 1, -1, -1):
        folder = {"name": tag_parts[i].strip(), "item": result}
        if i == 0 and tag_desc:
            folder["description"] = tag_desc
        result = [folder]

    return result


def collapse_single_child_folders(items):
    """
    If a folder has exactly one child that is also a folder (and no leaf items),
    merge them into a single folder with a combined name.
    """
    result = []
    for item in items:
        if "item" in item and "request" not in item:
            # It's a folder
            item["item"] = collapse_single_child_folders(item["item"])
            # Check if it has exactly one child that is also a folder
            while (len(item["item"]) == 1
                   and "item" in item["item"][0]
                   and "request" not in item["item"][0]):
                child = item["item"][0]
                item["name"] = item["name"] + "/" + child["name"]
                item["item"] = child["item"]
                if "description" not in item and "description" in child:
                    item["description"] = child["description"]
            result.append(item)
        else:
            result.append(item)
    return result


def merge_tag_folders(all_tag_results):
    """
    Merge folder trees from different tags that share the same top-level name.

    For example, tags 'Commerce/Actions' and 'Commerce/Layout' both produce
    a top-level 'Commerce' folder. This function merges them into one.
    """
    merged = {}
    order = []

    for tag_items in all_tag_results:
        for folder in tag_items:
            name = folder["name"]
            if name not in merged:
                merged[name] = folder
                order.append(name)
            else:
                # Merge items into existing folder
                merged[name]["item"].extend(folder.get("item", []))

    return [merged[name] for name in order]


def sort_items_recursive(items, do_format_names=True):
    """Alphabetically sort folders and requests at each level. Folders first.

    When do_format_names is True, also applies format_name() to folder names
    derived from URL path segments (camelCase -> Title Case).
    """
    folders = []
    requests = []
    for item in items:
        if "item" in item and "request" not in item:
            # Format folder names (but not request names, which come from Swagger summary)
            if do_format_names:
                # Handle collapsed folder names like "accounts/{customerId}"
                parts = item["name"].split("/")
                parts = [format_name(p) for p in parts]
                item["name"] = "/".join(parts)
            item["item"] = sort_items_recursive(item["item"], do_format_names)
            folders.append(item)
        else:
            requests.append(item)

    folders.sort(key=lambda x: x["name"].lower())
    requests.sort(key=lambda x: x["name"].lower())
    return folders + requests


# ---------------------------------------------------------------------------
# Collection Assembly
# ---------------------------------------------------------------------------


def build_collection(swagger, collection_name=None):
    """Assemble the full Postman Collection from a parsed Swagger dict."""
    info = swagger.get("info", {})
    definitions = swagger.get("definitions", {})
    global_consumes = swagger.get("consumes", ["application/json"])
    global_produces = swagger.get("produces", ["application/json"])

    # Build tag description lookup
    tag_descriptions = {}
    for tag in swagger.get("tags", []):
        tag_descriptions[tag["name"]] = tag.get("description", "")

    # Group operations by tag
    groups = group_operations_by_tag(swagger)

    # Build folder trees for each tag
    all_tag_results = []
    for tag_name, operations in groups.items():
        tag_folders = build_folder_for_tag(
            tag_name, operations, definitions, tag_descriptions,
            global_consumes, global_produces
        )
        all_tag_results.append(tag_folders)

    # Merge folders sharing the same top-level name
    top_level_items = merge_tag_folders(all_tag_results)

    # Sorting and name formatting are applied in main() so CLI flags are respected

    name = collection_name or "Postman REST APIs for Oracle CPQ"
    description = (
        "The latest up-to-date REST API collection for Oracle CPQ, "
        "compatible with Postman.\n\n"
        "Created by [Adrian Anderson](https://www.linkedin.com/in/aanderson1017/) "
        "| [YouTube - Better Human Applications](https://www.youtube.com/@BetterHumanApplications)"
    )

    collection = {
        "info": {
            "_postman_id": str(uuid.uuid4()),
            "name": name,
            "description": description,
            "schema": POSTMAN_SCHEMA,
        },
        "item": top_level_items,
        "auth": {
            "type": "basic",
            "basic": [
                {"key": "password", "value": "{{CPQ Password}}", "type": "string"},
                {"key": "username", "value": "{{CPQ UserName}}", "type": "string"},
            ],
        },
        "event": [
            {
                "listen": "prerequest",
                "script": {
                    "type": "text/javascript",
                    "packages": {},
                    "exec": [""],
                },
            },
            {
                "listen": "test",
                "script": {
                    "type": "text/javascript",
                    "packages": {},
                    "exec": [""],
                },
            },
        ],
        "variable": [],
    }

    return collection


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Swagger 2.0 JSON file to a Postman Collection v2.1.0 JSON file."
    )
    parser.add_argument("swagger_file", help="Path to the Swagger 2.0 JSON file")
    parser.add_argument(
        "-o", "--output",
        help="Output Postman collection JSON file path (default: <title>.postman_collection.json)",
    )
    parser.add_argument("-n", "--name", help="Collection name override")
    parser.add_argument(
        "--rest-version-value",
        default="rest/v19",
        help="Default RestVersion value hint (default: rest/v19)",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Do not sort folders and items alphabetically",
    )
    parser.add_argument(
        "--no-format-names",
        action="store_true",
        help="Do not convert camelCase/snake_case folder names to Title Case",
    )

    args = parser.parse_args()

    # Load Swagger
    with open(args.swagger_file, "r", encoding="utf-8") as f:
        swagger = json.load(f)

    # Validate it looks like Swagger 2.0
    swagger_version = swagger.get("swagger", "")
    if not swagger_version.startswith("2"):
        print(f"Warning: Expected Swagger 2.0, found version '{swagger_version}'", file=sys.stderr)

    # Build collection
    collection = build_collection(swagger, collection_name=args.name)

    # Optionally skip sorting / name formatting
    if not args.no_sort:
        do_format = not args.no_format_names
        collection["item"] = sort_items_recursive(collection["item"], do_format_names=do_format)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        title = collection["info"]["name"]
        safe_title = re.sub(r'[^\w\s-]', '', title).strip()
        output_path = os.path.join(os.path.dirname(args.swagger_file), f"{safe_title}.postman_collection.json")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(collection, f, indent=4, ensure_ascii=False)

    # Summary
    paths = swagger.get("paths", {})
    total_ops = sum(
        1 for path_item in paths.values()
        for method in ("get", "post", "put", "patch", "delete", "options", "head")
        if method in path_item
    )
    print(f"Converted {total_ops} operations from {len(paths)} paths")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
