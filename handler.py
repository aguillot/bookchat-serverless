import os
import sys
import tempfile
import zipfile
from typing import Any, Dict

import boto3
import magic
import tiktoken
from aws_lambda_powertools.utilities.parser import ValidationError, parse
from aws_lambda_powertools.utilities.typing import LambdaContext
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, UnstructuredEPubLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from loguru import logger

from schemas import EmbedInput, QueryInput

# outdated aws lambda image sqlite3 replacement
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


S3_BUCKET = os.environ.get("S3_BUCKET")
S3_UPLOADS_PREFIX = os.environ.get("S3_UPLOADS_PREFIX")
S3_VECTORS_PREFIX = os.environ.get("S3_VECTORS_PREFIX")

DEFAULT_COLLECTION_NAME = "bookchat"
OPENAI_TEMPERATURE = int(os.environ.get("OPENAI_TEMPERATURE", 0))
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

s3 = boto3.client("s3")


def get_file_loader(file_path: str):
    mime_type = magic.from_file(file_path, mime=True)
    match mime_type:
        case "application/pdf":
            return PyPDFLoader(file_path)
        case "application/epub+zip":
            return UnstructuredEPubLoader(file_path)
        case _:
            raise RuntimeError("unsupported file type")


def get_chat_history(messages_list: list) -> list:
    return [
        (messages_list[i], messages_list[i + 1])
        for i in range(0, len(messages_list), 2)
    ]


def embed(event: Dict[str, Any], context: LambdaContext) -> dict[str, Any]:
    logger.info(f"received event: {event}")
    try:
        input: EmbedInput = parse(event, model=EmbedInput)
    except ValidationError as e:
        logger.error(e)
        raise e

    logger.debug(input)

    key = f"{S3_UPLOADS_PREFIX}{str(input.book_id)}"

    tmpfile = tempfile.NamedTemporaryFile()

    try:
        s3.download_file(Bucket=S3_BUCKET, Key=key, Filename=tmpfile.name)
    except Exception as e:
        logger.error(e)
        raise e

    logger.info(f"downloaded book to path {tmpfile.name}")

    loader = get_file_loader(tmpfile.name)
    documents = loader.load()
    encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0

    for doc in documents:
        total_tokens += len(encoding.encode(doc.page_content))

    logger.debug(f"total tokens: {total_tokens}")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=input.openai_api_key)

    db_temp_dir = tempfile.TemporaryDirectory()

    db = Chroma.from_documents(
        texts,
        embeddings,
        collection_name=DEFAULT_COLLECTION_NAME,
        persist_directory=db_temp_dir.name,
    )

    db.persist()

    tmpfile.close()

    output_zip_file = f"/tmp/{input.book_id}.zip"

    with zipfile.ZipFile(output_zip_file, "w") as zipf:
        for root, dirs, files in os.walk(db_temp_dir.name):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path)

    db_temp_dir.cleanup()

    try:
        s3.put_object(
            Body=open(output_zip_file, "rb"),
            Bucket=S3_BUCKET,
            Key=f"{S3_VECTORS_PREFIX}{input.book_id}.zip",
        )
        s3.delete_object(Bucket=S3_BUCKET, Key=key)
    except Exception as e:
        logger.error(e)
        raise e

    logger.info(
        f"file uploaded to {S3_BUCKET}://{S3_VECTORS_PREFIX}{input.book_id}.zip"
    )

    return {"message": event}


def query(event: Dict[str, Any], context: LambdaContext) -> dict[str, Any]:
    logger.info(f"received event: {event}")
    try:
        input: QueryInput = parse(event, model=QueryInput)
    except ValidationError as e:
        logger.error(e)
        raise e

    tmp_db_path = f"/tmp/{input.book_id}"

    if os.path.exists(tmp_db_path):
        logger.info(f"Using cached vector db at {tmp_db_path}")
    else:
        logger.info(f"Downloading and extracting db into {tmp_db_path}")
        key = f"{S3_VECTORS_PREFIX}{str(input.book_id)}.zip"
        tmpfile = tempfile.NamedTemporaryFile()
        try:
            s3.download_file(Bucket=S3_BUCKET, Key=key, Filename=tmpfile.name)
        except Exception as e:
            logger.error(e)
            raise e
        with zipfile.ZipFile(tmpfile.name, "r") as zipf:
            zipf.extractall(tmp_db_path)
        tmpfile.close()

    # unzipped chromadb inside subfolders
    chroma_db_path = list(os.walk(tmp_db_path))[2][0]
    logger.debug(f"chromadb is located in: {chroma_db_path}")

    chat = ChatOpenAI(
        temperature=OPENAI_TEMPERATURE,
        model=OPENAI_MODEL,
        openai_api_key=input.openai_api_key,
    )

    embeddings = OpenAIEmbeddings(openai_api_key=input.openai_api_key)
    db = Chroma(
        collection_name=DEFAULT_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=chroma_db_path,
    )

    chain = ConversationalRetrievalChain.from_llm(
        chat, chain_type="stuff", retriever=db.as_retriever(), verbose=False
    )

    logger.debug(f"number of docs in db: {db._collection.count()}")

    with get_openai_callback() as cb:
        result = chain(
            {
                "question": input.chat_query,
                "chat_history": get_chat_history(input.chat_history),
            },
            return_only_outputs=True,
        )
        logger.debug(result["answer"])
        logger.debug(f"Total tokens: {cb.total_tokens}")
        # TODO: call to backend
        return {"input": event, "used_tokens": cb.total_tokens}
