from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers # because we are using qunatized model
from src.helper import *


B_INST, E_INST= "[INST]", "[/INST]"  # B_INST : begining of Instruction , E_INST : end of instruction
B_SYS, E_SYS = "<<SYS>>\n", "\n<<SYS>>\n\n" # B_sys : begining of system/default prompt, E_SYS : end of system/default prompt



instruction = "Convert the following text from English to Hindi: \n\n{text}"


SYSTEM_PROMPT=B_SYS+DEFAULT_SYSTEM_PROMPT+E_SYS # first build the system prompt
template=B_INST+SYSTEM_PROMPT+instruction+E_INST # now intergate the user prompt with system prompt


prompt = PromptTemplate(template=template, input_variables=["text"])


# here we are loading the quantized model using CTransformers

llm = CTransformers(model='model/llama-2-7b-chat.ggmlv3.q4_0.bin',
                    model_type='llama',
                    config={'max_new_tokens': 128,
                            'temperature': 0.01}
                   )

LLM_Chain=LLMChain(prompt=prompt, llm=llm)

print(LLM_Chain.run("How are you?"))

