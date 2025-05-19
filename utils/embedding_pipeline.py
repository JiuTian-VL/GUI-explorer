"""
在项目的根目录运行 python -m utils.embedding_pipeline 即可启动服务，然后在SiglipMultimodalEmbeddingPipeline的构造函数的device填写"server"。
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from PIL import Image
import gradio as gr
from transformers import AutoProcessor, AutoModel
from typing import Any, List, Union, Dict, Tuple
import numpy as np
from gradio_client import Client
import io
import base64
from utils.utils import resize_pil_image
MAX_BATCH_SIZE = 32  # 最大批处理大小，根据显存大小调整，50张最大512像素的图片大约需要2.5GB
MAX_IMAGE_SIZE = 512  # 最大图片大小，512x512，超过这个大小会自动保留比例缩放到最长边为512


class SiglipMultimodalEmbeddingPipeline:
    IMAGE_PREFIX = "<is_image>"
    SEPARATOR = "<sep>"

    def __init__(
        self,
        model_id: str = "google/siglip-so400m-patch14-384",
        device: str = "cuda",
        server_endpoint: str = "http://127.0.0.1:8765",
    ):
        self.device = device
        self.model_id = model_id
        self.client = None

        if self.device == "server":
            self.client = Client(server_endpoint, verbose=False)
        else:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
            if self.device == "cuda" and not torch.cuda.is_available():
                print("Warning: CUDA device requested but not available. Falling back to CPU.")
                self.device = "cpu"
                self.model.to(self.device)


    def __read_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    def __pillow_image_to_base64(self, pillow_image: Image.Image) -> str:
        img_byte_arr = io.BytesIO()
        # Resize the image if it's larger than MAX_IMAGE_SIZE
        if pillow_image.size[0] > MAX_IMAGE_SIZE or pillow_image.size[1] > MAX_IMAGE_SIZE:
            pillow_image = resize_pil_image(pillow_image, MAX_IMAGE_SIZE)
        pillow_image.convert("RGB").save(img_byte_arr, format="WEBP", quality=95)
        return base64.b64encode(img_byte_arr.getvalue()).decode()

    def embedding_images_or_texts(
        self, images_or_texts_str: str
    ) -> Union[List[float], List[List[float]]]:
        """
        This method is designed to be called by the Gradio server endpoint.
        It parses the special string format and calls the main embedding logic.
        """
        processed_input: Union[Image.Image, List[Image.Image], str, List[str]]
        is_single_input = self.SEPARATOR not in images_or_texts_str

        if images_or_texts_str.startswith(self.IMAGE_PREFIX):
            content_str = images_or_texts_str.removeprefix(self.IMAGE_PREFIX)
            if is_single_input:
                processed_input = Image.open(
                    io.BytesIO(base64.b64decode(content_str))
                ).convert("RGB")
            else:
                processed_input = [
                    Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
                    for img_b64 in content_str.split(self.SEPARATOR)
                    if img_b64 # Avoid empty strings if separator is at the end
                ]
        else: # Text input
            if is_single_input:
                processed_input = images_or_texts_str
            else:
                processed_input = [
                    text for text in images_or_texts_str.split(self.SEPARATOR)
                    if text # Avoid empty strings
                ]
        return self.__call__(processed_input)

    @torch.no_grad()
    def __call__(
        self, images_or_texts: Union[Image.Image, List[Image.Image], str, List[str], List[np.ndarray]]
    ) -> Union[List[float], List[List[float]]]:
        try:
            if self.device == "server":
                if self.client is None:
                    raise ConnectionError("Client not initialized. Set device='server' and provide server_endpoint.")
                return self.__call__server(images_or_texts)
            else:
                return self.__call__local(images_or_texts)
        except Exception as e:
            print(f"Error during embedding: {e}")
            if isinstance(images_or_texts, list) and len(images_or_texts) > 0:
                print(f"Problematic input type: {type(images_or_texts[0])}, count: {len(images_or_texts)}")
            else:
                print(f"Problematic input type: {type(images_or_texts)}")
            # For debugging, print a snippet of the input
            print(f"Input snippet: {str(images_or_texts)[:200]}")
            raise e

    def __call__server(
        self, images_or_texts: Union[Image.Image, List[Image.Image], str, List[str], List[np.ndarray]]
    ) -> Union[List[float], List[List[float]]]:
        serialized_input: str

        if isinstance(images_or_texts, str):  # Single text or image path
            if os.path.exists(images_or_texts):
                serialized_input = self.IMAGE_PREFIX + self.__read_image_to_base64(images_or_texts)
            else:
                serialized_input = images_or_texts
        elif isinstance(images_or_texts, Image.Image): # Single PIL Image
            serialized_input = self.IMAGE_PREFIX + self.__pillow_image_to_base64(images_or_texts)
        elif isinstance(images_or_texts, list):
            if not images_or_texts:
                return [] # Or raise error for empty list

            first_item = images_or_texts[0]
            if isinstance(first_item, str): # List of texts or image paths
                if os.path.exists(first_item): # Image paths
                    b64_images = [self.__read_image_to_base64(p) for p in images_or_texts]
                    serialized_input = self.IMAGE_PREFIX + self.SEPARATOR.join(b64_images) + self.SEPARATOR
                else: # Texts
                    serialized_input = self.SEPARATOR.join(images_or_texts) + self.SEPARATOR
            elif isinstance(first_item, Image.Image): # List of PIL Images
                b64_images = [self.__pillow_image_to_base64(img) for img in images_or_texts]
                serialized_input = self.IMAGE_PREFIX + self.SEPARATOR.join(b64_images) + self.SEPARATOR
            elif isinstance(first_item, np.ndarray): # List of numpy arrays (from Gradio image upload)
                pil_images = [Image.fromarray(img_arr).convert("RGB") for img_arr in images_or_texts]
                b64_images = [self.__pillow_image_to_base64(img) for img in pil_images]
                serialized_input = self.IMAGE_PREFIX + self.SEPARATOR.join(b64_images) + self.SEPARATOR
            else:
                raise ValueError(f"Unsupported list item type for server call: {type(first_item)}")
        else:
            raise ValueError(f"Unsupported input type for server call: {type(images_or_texts)}")

        rsp = self.client.predict(
            serialized_input,
            api_name="/embedding_images_or_texts",
        )
        return rsp

    @torch.no_grad()
    def __call__local(
        self, images_or_texts: Union[Image.Image, List[Image.Image], str, List[str], List[np.ndarray]]
    ) -> Union[List[float], List[List[float]]]:
        
        input_was_singular = False
        processed_input: Union[List[Image.Image], List[str]]

        # Normalize input to List[Image.Image] or List[str]
        if isinstance(images_or_texts, str):
            input_was_singular = True
            if os.path.exists(images_or_texts): # Image path
                img=Image.open(images_or_texts).convert("RGB")
                processed_input = [img]
            else: # Text
                processed_input = [images_or_texts]
        elif isinstance(images_or_texts, Image.Image):
            input_was_singular = True
            processed_input = [images_or_texts.convert("RGB")]
        elif isinstance(images_or_texts, list):
            if not images_or_texts:
                return []
            
            first_item = images_or_texts[0]
            if isinstance(first_item, str): # List of image paths or texts
                if os.path.exists(first_item): # Image paths
                    processed_input = [Image.open(p).convert("RGB") for p in images_or_texts]
                else: # Texts
                    processed_input = images_or_texts
            elif isinstance(first_item, Image.Image): # List of PIL Images
                processed_input = [img.convert("RGB") for img in images_or_texts]
            elif isinstance(first_item, np.ndarray): # List of numpy arrays (from Gradio)
                processed_input = [Image.fromarray(img_arr).convert("RGB") for img_arr in images_or_texts]
            else:
                raise ValueError(f"Unsupported list item type for local call: {type(first_item)}")
        else:
            raise ValueError(f"Unsupported input type for local call: {type(images_or_texts)}")

        # Determine if input is text or image
        is_text_input = isinstance(processed_input[0], str)
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(processed_input), MAX_BATCH_SIZE):
            batch = processed_input[i : i + MAX_BATCH_SIZE]
            features = []
            if is_text_input:
                inputs = self.processor(
                    text=batch, padding="max_length", truncation=True, return_tensors="pt"
                ).to(self.device)
                features = self.model.get_text_features(**inputs).cpu().tolist()
            else: # Image input
                for j in range(len(batch)):
                   if batch[j].size[0] > MAX_IMAGE_SIZE or batch[j].size[1] > MAX_IMAGE_SIZE:
                       batch[j] = resize_pil_image(batch[j], MAX_IMAGE_SIZE)
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                features = self.model.get_image_features(**inputs).cpu().tolist()
            all_embeddings.extend(features)

        return all_embeddings[0] if input_was_singular else all_embeddings


if __name__ == "__main__":
    os.environ["no_proxy"] = "localhost,127.0.0.1,::1" # Removed /8 as it's usually not needed for local
    print("Text and Image Embedding Service")

    # Determine device for the server hosting the model
    server_device = os.getenv("SERVER_EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {server_device}")

    print("Loading Multimodal Embedding Pipeline...")
    # This pipeline instance will run locally on the server
    multimodal_pipeline = SiglipMultimodalEmbeddingPipeline(
        model_id="google/siglip-so400m-patch14-384", # or "marco/mcdse-2b-v1"
        device=server_device,
    )
    print("Multimodal Embedding Pipeline Loaded.")

    # --- Gradio UI Helper Functions ---
    def embed_texts_gradio(text_input_str: str):
        if not text_input_str.strip():
            return [], {"error": "Input is empty"}
        texts = [t.strip() for t in text_input_str.splitlines() if t.strip()]
        if not texts:
            return [], {"error": "No valid text lines found"}
        
        # Prepare string for embedding_images_or_texts method
        # This simulates what __call__server would do if it were calling this server
        # If texts is a list: "text1<sep>text2<sep>"
        # If texts is single: "text1"
        if len(texts) == 1:
            api_input_str = texts[0]
        else:
            api_input_str = SiglipMultimodalEmbeddingPipeline.SEPARATOR.join(texts) + SiglipMultimodalEmbeddingPipeline.SEPARATOR
        
        embeddings = multimodal_pipeline.embedding_images_or_texts(api_input_str)
        shape = np.array(embeddings).shape
        return embeddings, {"shape": str(shape)}

    def embed_single_image_gradio(image_input: Image.Image): # Gradio gives PIL Image if type="pil"
        if image_input is None:
            return [], {"error": "No image uploaded"}
        
        # Prepare string for embedding_images_or_texts method
        # This simulates what __call__server would do
        b64_image = multimodal_pipeline._SiglipMultimodalEmbeddingPipeline__pillow_image_to_base64(image_input)
        api_input_str = SiglipMultimodalEmbeddingPipeline.IMAGE_PREFIX + b64_image
        
        embeddings = multimodal_pipeline.embedding_images_or_texts(api_input_str)
        shape = np.array(embeddings).shape
        return embeddings, {"shape": str(shape)}

    def embed_multiple_images_gradio(image_files: List[str]): # Gradio gives list of filepaths if type="filepath"
        if not image_files:
            return [], {"error": "No images uploaded"}
        
        # Prepare string for embedding_images_or_texts method
        b64_images = [multimodal_pipeline._SiglipMultimodalEmbeddingPipeline__read_image_to_base64(f.name) for f in image_files]
        api_input_str = (
            SiglipMultimodalEmbeddingPipeline.IMAGE_PREFIX + 
            SiglipMultimodalEmbeddingPipeline.SEPARATOR.join(b64_images) + 
            SiglipMultimodalEmbeddingPipeline.SEPARATOR
        )
        
        embeddings = multimodal_pipeline.embedding_images_or_texts(api_input_str)
        shape = np.array(embeddings).shape
        return embeddings, {"shape": str(shape)}

    print("Launching Gradio...")
    with gr.Blocks() as demo:
        gr.Markdown("### Text and Image Embedding Service")

        with gr.Tab("Text Embedding"):
            text_input = gr.Textbox(
                lines=7, label="Input Text(s)", info="Enter one text per line for multiple texts."
            )
            with gr.Row():
                text_button = gr.Button("Generate Text Embedding(s)")
            text_output_json = gr.JSON(label="Embedding Vector(s)")
            text_output_shape = gr.JSON(label="Output Shape")
            text_button.click(
                embed_texts_gradio,
                inputs=text_input,
                outputs=[text_output_json, text_output_shape],
                api_name="embed_texts" # For client.predict
            )
        
        with gr.Tab("Single Image Embedding"):
            # For single image, type="pil" is convenient to get PIL.Image directly
            single_image_input = gr.Image(type="pil", label="Upload Single Image")
            with gr.Row():
                single_image_button = gr.Button("Generate Image Embedding")
            single_image_output_json = gr.JSON(label="Embedding Vector")
            single_image_output_shape = gr.JSON(label="Output Shape")
            single_image_button.click(
                embed_single_image_gradio,
                inputs=single_image_input,
                outputs=[single_image_output_json, single_image_output_shape],
                api_name="embed_single_image"
            )

        with gr.Tab("Multiple Images Embedding"):
            # For multiple images, type="filepath" or type="bytes"
            # Using file_count="multiple" with gr.File
            multi_image_input = gr.File(
                label="Upload Multiple Images",
                file_count="multiple",
                file_types=["image"], # e.g., .png, .jpg, .jpeg, .webp
                type="filepath" # Gives list of tempfile._TemporaryFileWrapper objects, use .name for path
            )
            with gr.Row():
                multi_image_button = gr.Button("Generate Image Embeddings")
            multi_image_output_json = gr.JSON(label="Embedding Vectors")
            multi_image_output_shape = gr.JSON(label="Output Shape")
            multi_image_button.click(
                embed_multiple_images_gradio,
                inputs=multi_image_input,
                outputs=[multi_image_output_json, multi_image_output_shape],
                api_name="embed_multiple_images"
            )
        
        # This is the original endpoint that __call__server targets
        # It's kept for compatibility if you have clients using it directly
        # but the UI above uses more specific helper functions for clarity.
        gr.Textbox(
            label="Internal API input (for client calls)", 
            visible=False # Hide from UI unless debugging
        ).submit(
            multimodal_pipeline.embedding_images_or_texts,
            inputs=gr.Textbox(), # This needs to be a Textbox to accept the string
            outputs=gr.JSON(), # Output should be JSON for vectors
            api_name="embedding_images_or_texts" # This matches the client
        )


    demo.queue().launch(share=False, server_name="0.0.0.0", server_port=8765)