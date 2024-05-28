# LangChain

## Install the OpenAI Python library

From the terminal / command line, run:

```bash
pip install --upgrade openai
```

### Set up OpenAI API key

Create `.env` file in the root of your project and add your OpenAI API key:

```bash
OPENAI_API_KEY=your-api-key
```

Run the following command to install the `python-dotenv` package:

```bash
pip install python-dotenv
```

Add the following code to your Python script to load the API key from the `.env` file:

```python
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key=os.environ.get("OPENAI_API_KEY")
```

### Test your API key

Create a new Python script and add the following code to test your API key:

```python
from openai import OpenAI

client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY")
)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)
```

## Install LangChain

To install the LangChain Python library, run the following command in your terminal or command line:

```bash
pip install langchain
```

To use OpenAI with LangChain, install the OpenAI Python library that integrates with LangChain:

```bash
pip install langchain-openai
```

### Set up OpenAI API key with LangChain

Create a Python script with the following code to run your first LangChain program:

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv('.env')

llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

llm.invoke("how can langsmith help with testing?")
```

### Create a chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv('.env')

llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

response = chain.invoke("how can langsmith help with testing?")
```