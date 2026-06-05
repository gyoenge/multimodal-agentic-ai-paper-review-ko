project = "VLM Paper Review KO"
author = "gyoenge"
copyright = "2024, gyoenge"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
]

autosectionlabel_prefix_document = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "furo"
html_title = "VLM Paper Review KO"
html_static_path = ["_static"]

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#1d4ed8",
        "font-stack": "'Noto Sans KR', 'Pretendard', -apple-system, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'Fira Code', monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#93c5fd",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

html_css_files = ["custom.css"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
