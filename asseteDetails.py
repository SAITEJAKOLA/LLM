import requests
import os
from dotenv import load_dotenv
from simple_local_arg import retrieve_answers_with_llm_model

load_dotenv()
def fetch_asset_details(asset_name):
    api_key = os.environ.get("LYNX_API_KEY")
    api_url = f"https://api.dev.fleet.lynx.carrier.io/v1/asset-snapshots?assetNames={asset_name}"
    headers = {
        "x-lynx-api-key":  api_key
    }

    try:
        response = requests.get(api_url, headers=headers)
        print(response)
        response.raise_for_status()
        asset_data = response.json()
        print(asset_data)
        if asset_data and asset_data['data']:
            asset_details = f"Asset Name: {asset_name}\n"
            for key, value in asset_data['data'][0].items():
                if isinstance(value, dict):
                    asset_details += f"\n{key}:\n"
                    for inner_key, inner_value in value.items():
                        asset_details += f"  {inner_key}: {inner_value}\n"
                elif isinstance(value, list):
                    asset_details += f"\n{key}:\n"
                    for item in value:
                        if isinstance(item, dict):
                            asset_details += "  {\n"
                            for inner_key, inner_value in item.items():
                                asset_details += f"    {inner_key}: {inner_value}\n"
                            asset_details += "  }\n"
                        else:
                            asset_details += f"  {item}\n"
                else:
                    asset_details += f"{key}: {value}\n"
            prompt = f"Using the provided asset_details context, extract and display the following specific information in a tabular format: assetName, powerStatus, truStatus, and address. If any of these details are missing from the given data, create a plausible context using the available asset_details. Here is the data: {asset_details}"
            context = asset_details
            return retrieve_answers_with_llm_model(prompt, context)
        else:
            return f"No asset details found for {asset_name}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching asset details: {e}"

# # Example usage
# asset_details = fetch_asset_details("SIMULATED-DHI-ASSET-DEV-192")
# print(asset_details)
