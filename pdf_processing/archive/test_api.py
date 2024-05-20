import os
import re
import json
import base64
import requests
import time
import urllib.parse

API_KEY = os.environ.get('API_KEY')  # Match the environment variable name to the name you used in the .env file
API_VERSION = os.environ.get('API_VERSION')
RESOURCE_ENDPOINT = os.environ.get('RESOURCE_ENDPOINT')

# The deployment choice uniquely determines the underlying model, so there is no need to separately test models
# Add future/delete deprecated deployments to/from the lists below
completions_deployments = [
    'text-davinci-003',
    'gpt-35-turbo-instruct',
]
embeddings_deployments = [
    'text-embedding-ada-002', 
]
chat_deployments = [
    'gpt-35-turbo',
    'gpt-35-turbo-0301',
    'gpt-4',
    'gpt-35-turbo-16K',
    'gpt-4-32K',
    'gpt-4-turbo-128k',
]  

# These are configurable parameters for managing re-attempts for API calls
RETRY_SECS = 15  # Seconds between attempts
MAX_RETRIES = 5  # Max number of re-attempts

error_msg = "\nProvided your configuration parameters (API_KEY, API_VERSION, RESOURCE_ENDPOINT, deployment name) are valid, the majority of errors you may encounter with this code are attributable to temporary issues such as Azure server outages or other users who have triggered shared API rate limits for a given deployment. Please try again in a few minutes. However, if you receive a 401 Unauthorized access error, while your API key may have the correct length, most likely it is not a valid key for some other reason. In that event, please open a ticket with the Versa team at versa@ucsf.edu to review the key.\n"


# The next three functions are for testing your environment variables
def test_key():
    assert API_KEY is not None and API_KEY.strip() != "", "API Key is missing or empty"
    try:
        redacted_key = API_KEY[0] + "*" * (len(API_KEY) - 3) + API_KEY[-2:]
        base64.b64decode(API_KEY)
        print(f"API Key is a valid base64 string with length={len(API_KEY)}: {redacted_key}. Although the key is correctly formatted, it is possible that the key may be invalid for some other reason. The remaining tests are needed to fully validate the key.")
    except Exception as e:
        assert False, f"API Key is not a valid base64 string: {redacted_key} " + str(e)

def test_version():
    assert API_VERSION is not None and API_VERSION.strip() != "", "API Version is missing or empty"
    pattern = r'\d{4}-\d{2}-\d{2}'  # matches four digits-two digits-two digits
    assert re.fullmatch(pattern,
                        API_VERSION) is not None, f"API version has invalid format, it should be like: yyyy-mm-dd: {API_VERSION}"
    print(f"API version has valid format: yyyy-mm-dd: {API_VERSION}")

def test_endpoint():
    assert RESOURCE_ENDPOINT is not None and RESOURCE_ENDPOINT.strip() != "", "Resource endpoint is missing or empty"
    url = urllib.parse.urlparse(RESOURCE_ENDPOINT)
    assert all([url.scheme, url.netloc]), f"Resource endpoint is not a valid URL: {RESOURCE_ENDPOINT}"
    print(f"Resource endpoint is a valid URL: {RESOURCE_ENDPOINT}")


# The next three functions will use your environment variables to test the API
# For code testing purposes, passing True as the argument for simulate_api_error will run the test against an invalid deployment name
def test_chat_completions(simulate_api_error=False):
    deployments = ['fake_deployment'] if simulate_api_error else chat_deployments
    
    for deployment_id in deployments:
        print(f"\nTesting chat completions for deployment: {deployment_id}")
        url = f'{RESOURCE_ENDPOINT}/openai/deployments/{deployment_id}/chat/completions?api-version={API_VERSION}'
        prompt = 'Hello, how are you today?'

        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}]
        })
        headers = {'Content-Type': 'application/json', 'api-key': API_KEY}

        retries = 0

        while True:
            try:
                response = post_request(url, headers, body)

                print('User: ', prompt)
                print('Response: ', json.loads(response.text).get('choices')[0].get('message').get('content'))

                break
            except Exception as e:
                retries = exception_code(retries, deployment_id, e)

def test_completions(simulate_api_error=False):
    deployments = ['fake_deployment'] if simulate_api_error else completions_deployments
        
    for deployment_id in deployments:
        print(f"\nTesting completions for deployment: {deployment_id}")
        completions_url = f"{RESOURCE_ENDPOINT}/openai/deployments/{deployment_id}/completions?api-version={API_VERSION}"
        prompt = 'The rain in Spain'
        body = json.dumps({
            "prompt": prompt,
            "max_tokens": 30,  # Limit the response
        })
        headers = {
            'Content-Type': 'application/json',
            'api-key': API_KEY
        }

        retries = 0

        while True:
            try:
                response = post_request(completions_url, headers, body)

                print('User: ', prompt)
                print('Response: ', json.loads(response.text).get('choices')[0].get('text'))
                break
            except Exception as e:
                retries = exception_code(retries, deployment_id, e)

def test_embeddings(simulate_api_error=False):
    deployments = ['fake_deployment'] if simulate_api_error else embeddings_deployments

    for deployment_id in deployments:
        print(f"\nTesting embeddings for deployment: {deployment_id}")
        embeddings_url = f"{RESOURCE_ENDPOINT}/openai/deployments/{deployment_id}/embeddings?api-version={API_VERSION}"
        body = json.dumps({
            "input": "This is test string to embed",
        })
        headers = {
            'Content-Type': 'application/json',
            'api-key': API_KEY
        }

        retries = 0

        while True:
            try:
                response = post_request(embeddings_url, headers, body)

                embedding_len = len(json.loads(response.text)['data'][0]['embedding'])
                print('Embedding received from API')

                if deployment_id == 'text-embedding-ada-002':
                    assert_len = 1536
                else:
                    raise ValueError(f'Deployment {deployment_id} not supported for validation. Check code')

                assert embedding_len == assert_len, f"Test failed for deployment: {deployment_id}, Response status code: {response.status_code}, Response: {response.text}"

                break
            except Exception as e:
                retries = exception_code(retries, deployment_id, e)

# These two functions are helper functions
def post_request(url, headers, body):
    response = requests.post(url, headers=headers, data=body)
        
    response.raise_for_status()
    return response

def exception_code(retries, deployment_id, e):
    if retries >= MAX_RETRIES:
        print(f'Failed attempt {retries+1} of {MAX_RETRIES+1}.')
        print(error_msg)
        
        assert False, f"Test failed for deployment: {deployment_id}, Error received: {e}"
    else:
        print(f'Failed attempt {retries+1} of {MAX_RETRIES + 1}. Waiting {RETRY_SECS} secs before next attempt...')
        
    retries += 1
    time.sleep(RETRY_SECS)
    
    return retries


# First perform some basic validation of our environment variables. Failure of any of these first 3 tests will cause all subsequent tests to fail as well.
test_key()
test_version()
test_endpoint()

# Next, run the API tests once the above three tests have passed
test_chat_completions()  # Responds as an AI assistant (This is what most users will need)
test_completions()  # Continues a thought or sentence in the prompt
test_embeddings()  # Validates a properly formed embedding

print('Tests completed')