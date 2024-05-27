import sys
import yaml

from openai import OpenAI, AzureOpenAI

api_key = yaml.safe_load(open("config.yaml"))["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

config = yaml.safe_load(open("config.yaml"))
azure_client = AzureOpenAI(
    azure_endpoint=config["base_url"],
    api_key=config["azure_api_key"],
    api_version=config["api_version"],
)


def gen_openai_completion(prompt: str, model="gpt-4o", visualize_stream=True) -> str:
    if visualize_stream:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        res_str = ""
        for chunk in res:
            delta = chunk.choices[0].delta.content
            if delta is None:
                break

            res_str += delta
            sys.stdout.write(delta)
            sys.stdout.flush()

        return res_str
    else:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        return str(res.choices[0].message.content)


def get_response(prompt: str):
    messages = [{"role": "user", "content": prompt}]
    response = azure_client.chat.completions.create(
        model="gpt4_large",
        messages=messages,
    )
    return f"{response.choices[0].message.content}"