from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.services.document_service import DocumentService
from app.utils.logger import logger
from app.exceptions.custom_exceptions import InvalidFileTypeException, DocumentProcessingException
import os

app = FastAPI()

document_service = DocumentService()


@app.post("/upload/")
async def upload_documents(files: list[UploadFile] = File(...)):
    """
       Uploads and processes multiple PDF documents.

       Args:
           files (list[UploadFile]): A list of PDF files to upload.

       Returns:
           JSONResponse: A success message if the documents are processed successfully.

       Raises:
           HTTPException: If the file type is invalid or an error occurs during processing.
       """

    try:
        for file in files:
            if file.content_type != "application/pdf":
                raise InvalidFileTypeException("Only PDF files are allowed.")

            file_path = f"uploads/{file.filename}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "wb") as f:
                f.write(file.file.read())

            document_service.process_document(file_path)

        return JSONResponse(content={"message": "Documents uploaded and processed successfully."}, status_code=200)

    except InvalidFileTypeException as e:
        logger.error(str(e))
        raise HTTPException(status_code=400, detail=str(e))

    except DocumentProcessingException as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")




@app.post("/query/")
async def query_document(querys: list[str]):
    """
    Queries the most relevant document collection and generates a response using an LLM.

    Args:
       query (str): The user's query.

    Returns:
       JSONResponse: The LLM-generated response.

    Raises:
       HTTPException: If an error occurs during query processing.
    """

    results = {}
    try:
        for query in querys:
            collection = document_service.select_collection(query)
            if not collection:
                results[query] = "No relevant answer found."
                continue
            GOOD_SCORE_THRESHOLD = 0.5
            docs = collection.similarity_search_with_score(query=query, k=3)
            # Filter and display results with a good score
            for doc, score in docs:
                if score >= GOOD_SCORE_THRESHOLD:
                    print(">>>>> content>>",doc.page_content)
                    result = document_service.parsing_to_llm(query, doc.page_content)
                    results[query] = result
                # else:
                #     results[query] = "No relevant answer found."
        return JSONResponse(content={"results": results}, status_code=200)

    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the query.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)