import ast

with open('app.py') as f:
    source = f.read()

tree = ast.parse(source)

funcs = {}
classes = {}
assignments = {}

for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        funcs[node.name] = ast.get_source_segment(source, node)
    elif isinstance(node, ast.ClassDef):
        classes[node.name] = ast.get_source_segment(source, node)
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = ast.get_source_segment(source, node)

import json
with open('app_ast.json', 'w') as f:
    json.dump({"funcs": list(funcs.keys()), "assignments": list(assignments.keys())}, f)

