# %%
import openai
import yaml

# %%
with open("config.yaml", "r") as file:
    data = yaml.safe_load(file)

# Print the content of the YAML file
print(data)
# %%

# %%
