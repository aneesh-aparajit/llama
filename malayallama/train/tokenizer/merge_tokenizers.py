# extracted from Malayalam-LLaMA
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm

llama_tokenizer_dir = "meta-llama/Llama-2-7b-chat-hf"
malayalam_sp_model_file = "../../checkpoints/malayalam-20k-1.2mil/malayalam-20k.model"
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
malayalam_sp_model = spm.SentencePieceProcessor()
malayalam_sp_model.Load(malayalam_sp_model_file)

llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
malayalam_spm = sp_pb2_model.ModelProto()
malayalam_spm.ParseFromString(malayalam_sp_model.serialized_model_proto())


print(len(llama_tokenizer), len(malayalam_sp_model))
print(llama_tokenizer.all_special_tokens)
print(llama_tokenizer.all_special_ids)
print(llama_tokenizer.special_tokens_map)

# add tokens
llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
len(llama_spm_tokens_set)

for p in malayalam_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
print(len(llama_spm.pieces))

# save model
output_sp_dir = "../../checkpoints/merged_tokenizer_sp"
output_hf_dir = "../../checkpoints/merged_tokenizer_hf"  # the path to save Malayalam-LLaMA tokenizer
os.makedirs(output_sp_dir, exist_ok=True)

with open(output_sp_dir + "/malayalam_llama.model", "wb") as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + "/malayalam_llama.model")

tokenizer.save_pretrained(output_hf_dir)
print(f"Malayalam-LLaMA tokenizer has been saved to {output_hf_dir}")

# test
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
malayalam_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.special_tokens_map)
text = """ജിപ്സീസ് എന്ന ഒരിടത്തും നിൽക്കാത്ത ഒരു ജനത അന്നു അക്കാലത്തും ആ റഷ്യയിൽ ഉണ്ടായിരുന്നു എന്നത് കാരമസോവ് ബ്രദേഴ്സ് വായിച്ചവർക്ക് അറിയാം. വേശ്യകളെ പുനരധിവസിപ്പിക്കാൻ ധാരാളം ശ്രമങ്ങൾ നടന്നിരുന്നു. മാത്രമല്ല വിശുദ്ധ വേശ്യകളെ ചിത്രീകരിച്ചുകൊണ്ട് അക്കാലത്തെ ധാരാളം സാഹിത്യകൃതികൾ വന്നിരുന്നു. 1863ൽ നിക്കോളായി ചെർണിഷേവ്സ്കിയുടെ വാട്ട് ഈ റ്റു ബി ഡൺ? എന്ന നോവൽ ഉദാഹരണം. ദൊസ്റ്റോയോവ്സ്കി വിക്റ്റർ ഹ്യൂഗോയുടെ ഒരു ആരാധകൻ ആയിരുന്നു എന്ന് ജോസഫ് ഫ്രാങ്ക്, ദൊസ്റ്റോയോവ്സ്കിയുടെ ജീവചരിത്രകാരൻ പറയുന്നു. ഹ്യൂഗോയുടെ സോഷ്യൽ ഹ്യുമാനിറ്റേറിയനിസം ദൊസ്റ്റോയോവ്സ്കിയെ സ്വാധീനിച്ചിരുന്നു. പുഷ്കിൻ ആയിരുന്നു മറ്റൊരു ഫേവറൈറ്റ് റൈറ്റർ.
The primary use of LLaMA is research on large language models, including"""
print("Test text:\n", text)
print(f"Tokenized by LLaMA tokenizer:\n{llama_tokenizer.tokenize(text)}")
print(f"+++++")
print(
    f"Tokenized by Malayalam-LLaMA tokenizer:\n{malayalam_llama_tokenizer.tokenize(text)}"
)
