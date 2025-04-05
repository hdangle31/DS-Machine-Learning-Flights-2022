import os
import re

# Read the README.md file
with open('README.md', 'r', encoding='utf-8') as f:
    markdown_text = f.read()

# Basic conversion of headings
html = markdown_text
html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
html = re.sub(r'^#### (.*?)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)

# Convert images
html = re.sub(r'!\[(.*?)\]\((.*?)\)', r'<img src="\2" alt="\1" style="max-width:100%;">', html)

# Convert italics (simple cases)
html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)

# Convert bold (simple cases)
html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)

# Convert links (simple cases)
html = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', html)

# Convert code blocks (very basic)
html = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)

# Convert inline code
html = re.sub(r'`(.*?)`', r'<code>\1</code>', html)

# Convert paragraphs (simple approach)
paragraphs = html.split('\n\n')
html = '\n'.join([f'<p>{p}</p>' if not p.startswith('<') else p for p in paragraphs])

# Add basic styling
html_with_style = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flight Cancellation Prediction</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        code {{
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 4px;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        .image-caption {{
            text-align: center;
            font-style: italic;
            margin-top: -15px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    {html}
</body>
</html>
"""

# Save the HTML file
with open('README.html', 'w', encoding='utf-8') as f:
    f.write(html_with_style)

print(f"HTML file created: {os.path.abspath('README.html')}")