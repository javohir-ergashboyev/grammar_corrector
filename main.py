from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

corrector = pipeline(
              'text2text-generation',
              'pszemraj/grammar-synthesis-small',
              )

class InputText(BaseModel):
    text: str

@app.post("/grammar-correction")
async def grammar_correction(input_text: InputText):
    input_text = input_text.text
    
    outputs = corrector(input_text)
    
    
    return {"corrected_text": outputs[0]['generated_text']}
# @app.post("/coherent-text")
# async def coherent_form(input_text: InputText):
#     input_text = 'make this text more coherent:'+input_text.text
#     input_ids = tokenizer(input_text, return_tensors='tf').input_ids
    
#     outputs = model.generate(input_ids, max_length=256)
    
#     corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     return {"coherent": corrected_text}
