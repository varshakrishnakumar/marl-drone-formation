import os
import re

def remove_comments_from_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        # Remove full-line comments and inline comments
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        # Remove inline comments, but avoid removing '#' in strings
        if '#' in line:
            # Only remove if not inside a string
            parts = re.split(r'(?<!["\'])#', line)
            line = parts[0].rstrip() + '\n'
        new_lines.append(line)
    with open(filepath, 'w') as f:
        f.writelines(new_lines)

rl_folder = 'rl/'
for root, dirs, files in os.walk(rl_folder):
    for file in files:
        if file.endswith('.py'):
            remove_comments_from_file(os.path.join(root, file))