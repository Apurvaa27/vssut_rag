"""Read and chunk supported local documents."""

from pathlib import Path
import re

from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".php"}


def php_to_markdown(php_path: Path) -> Path:
    """Convert PHP/HTML files to a Markdown file next to the source."""
    php_path = Path(php_path)
    if not php_path.exists():
        raise FileNotFoundError(f"File not found: {php_path}")

    source = php_path.read_text(encoding="utf-8", errors="replace")
    if re.search(
        r"<\s*(?:!doctype|html|body|main|section|article|div|h[1-6]|p|ul|ol|li|table|title)\b",
        source,
        re.I,
    ):
        soup = BeautifulSoup(source, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        # Preserve text from div/span-based layouts too; selecting only p/h/li
        # elements silently dropped much of the content on templated PHP pages.
        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            text = " ".join(element.get_text(" ", strip=True).split())
            level = int(element.name[1])
            element.replace_with(f"\n{'#' * level} {text}\n" if text else "\n")
        for element in soup.find_all("li"):
            text = " ".join(element.get_text(" ", strip=True).split())
            element.replace_with(f"\n- {text}\n" if text else "\n")
        for element in soup.find_all(
            ["p", "div", "section", "article", "main", "tr", "br"]
        ):
            element.insert_before("\n")
            element.insert_after("\n")

        content = soup.get_text(" ", strip=True)
        content = re.sub(r"[ \t]*\n[ \t]*", "\n", content)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()
    else:
        content = f"```php\n{source.rstrip()}\n```"

    markdown_path = php_path.with_suffix(".md")
    markdown = f"# {php_path.stem}\n\n{content}\n"
    # Avoid changing the generated file's timestamp on every Streamlit rerun.
    if not markdown_path.exists() or markdown_path.read_text(encoding="utf-8") != markdown:
        markdown_path.write_text(markdown, encoding="utf-8")
    return markdown_path


def load_documents(data_dir: Path) -> list[Document]:
    """Recursively load supported files and convert PHP files before loading."""
    data_dir.mkdir(parents=True, exist_ok=True)
    documents: list[Document] = []

    for php_path in sorted(data_dir.rglob("*")):
        if php_path.is_file() and php_path.suffix.lower() == ".php":
            php_to_markdown(php_path)

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        if path.suffix.lower() == ".pdf":
            loaded = PyMuPDFLoader(str(path)).load()
        else:
            loaded = TextLoader(
                str(path), encoding="utf-8", autodetect_encoding=True
            ).load()

        for document in loaded:
            php_source = path.with_suffix(".php") if path.suffix.lower() == ".md" else None
            source_path = php_source if php_source and php_source.exists() else path
            document.metadata["source"] = source_path.relative_to(data_dir).as_posix()
            document.metadata["file_type"] = source_path.suffix.lower().lstrip(".")
        documents.extend(loaded)

    return documents


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)
