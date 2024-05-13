
from openai import AzureOpenAI
import os

from PIL import Image
import fitz  # PyMuPDF
import mimetypes

import base64
from mimetypes import guess_type

aoai_api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
aoai_api_key= os.getenv("AZURE_OPENAI_API_KEY")
aoai_deployment_name = 'gpt-4-vision' # your model deployment name for GPT-4V
aoai_api_version = '2024-02-15-preview' # this might change in the future

MAX_TOKENS = 2000

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def understand_image_with_gptv(image_path, caption):
    """
    Generates a description for an image using the GPT-4V model.

    Parameters:
    - api_base (str): The base URL of the API.
    - api_key (str): The API key for authentication.
    - deployment_name (str): The name of the deployment.
    - api_version (str): The version of the API.
    - image_path (str): The path to the image file.
    - caption (str): The caption for the image.

    Returns:
    - img_description (str): The generated description for the image.
    """
    client = AzureOpenAI(
        api_key=aoai_api_key,  
        api_version=aoai_api_version,
        base_url=f"{aoai_api_base}/openai/deployments/{aoai_deployment_name}"
    )

    data_url = local_image_to_data_url(image_path)
    response = client.chat.completions.create(
                model=aoai_deployment_name,
                messages=[
                    { "role": "system", "content": "You are a helpful assistant." },
                    { "role": "user", "content": [  
                        { 
                            "type": "text", 
                            "text": f"Describe this image (note: it has image caption: {caption}):" if caption else "Describe this image:"
                        },
                        { 
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            }
                        }
                    ] } 
                ],
                max_tokens=2000
            )
    img_description = response.choices[0].message.content
    return img_description


caption = 'Resistance to Anti-estrogens and KDM5i in MCF7 Cells (A) Cellular viability after treatment with C70 and C49, fulvestrant, or tamoxifen in parental and cells with acquired resistance to the indicated agents. Error bars represent SD, n = 6. (B) Bar graph depicting percentage of unique barcodes in FULVR and TAMR relative to parental MCF7 cells at same passage. (C) Pie chart depicting percentage of barcodes overlapping between MCF7 and FULVR/TAMR cells. (D) Bar graph depicting percentage of total barcodes shared among all replicates in each of the indicated cell populations. (E) Pie chart depicting percentage of barcodes overlapping between FULVR and TAMR. (F) Bar graph depicting percentage of unique barcodes in C70R and C49R relative to MCF7 cells at same passage. (G) Pie chart depicting percentage of barcodes overlapping between MCF7 and C70R/C49R cells.(H) Panels show model-predicted percentages of total barcodes shared by quadruplicates after simulation for different mutation probabilities (m) and seeded fractions of pre-existing resistant barcodes (r) in the treatment with the indicated inhibitors compared with the same statistic from the experimental data (horizontal line). The growth rates in simulations were based on experimental data. (I) Mutated genes detected in resistant but not in MCF7 cells. Colors and stars indicate the type of mutations and significance of downstream GSEA in the corresponding resistant cell lines, respectively. The significance of downstream GSEA represents the downstream genes of mutations are significantly enriched in up-/downregulated genes in the corresponding resistant cell lines. See also FigureS6and TableS6.'

image_path = "/Users/kkumbier/Desktop/fig6.png"
url = local_image_to_data_url(image_path)

result = understand_image_with_gptv(image_path, caption)