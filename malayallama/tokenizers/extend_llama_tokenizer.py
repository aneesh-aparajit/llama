from transformers import LlamaTokenizerFast
from tqdm import tqdm

tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b-chat-hf")


def generate_train_corpus():
    with open("../../data/CulturaX/text/malayalam_pretraining_corpus.txt", "r") as f:
        lines = f.readlines()
        length = len(lines)
        print(f"Loaded lines... Length: {length}")
        for line in tqdm(lines, total=length):
            yield line


for ix, t in enumerate(generate_train_corpus()):
    print(ix, " --> ", t)
    if ix == 10:
        break

print(f"Length of Old Tokenizer: {len(tokenizer)}")

print(f"Training Tokenizer...")
new_tokenizer = tokenizer.train_new_from_iterator(generate_train_corpus(), 52000)
new_tokenizer.save_pretrained("../../checkpoints/malayalam-52k/")

text = "കാസര്‍കോട്: സമൂഹ മാധ്യമത്തിലൂടെ ഡോക്‌ടർ ചമഞ്ഞ് ഏഴുലക്ഷം രൂപ തട്ടിയ കേസിൽ ഉത്തർപ്രദേശ് സ്വദേശി അറസ്‌റ്റിൽ. ഉത്തർപ്രദേശ് ബറേലി സ്വദേശി മുഹമ്മദ്‌ ഷാരിക്കിനെ(19)യാണ് കാസർകോട് സൈബർ ക്രൈം പൊലീസ് അറസ്‌റ്റ് ചെയ്‌തത്. സമൂഹ മാധ്യമത്തിലൂടെ ഡോക്‌ടര്‍ ചമഞ്ഞ് മായിപ്പാടി സ്വദേശിനിയിൽ നിന്നാണ് ഇയാള്‍ ഏഴുലക്ഷം രൂപ തട്ടിയത്.ഒരു കുഞ്ഞുണ്ടെങ്കിലും രണ്ടാമത്തെ കുഞ്ഞുണ്ടാകാൻ വൈകിയതിനെ തുടർന്നാണ് സുഹൃത്തിന്‍റെ നിർദേശ പ്രകാരം യുകെയിലെന്ന് അവകാശപ്പെട്ടിരുന്ന ഡോക്‌ടറെ മൂന്നു മാസം മുമ്പ് യുവതി പരിചയപ്പെട്ടത്. കാര്യങ്ങൾ ചോദിച്ചറിയുകയും ഗർഭധാരണ മരുന്നുകൾ കൈവശമുണ്ടെന്നും ഇയാൾ യുവതിയെ വിശ്വസിപ്പിച്ചു. ഇൻസ്‌റ്റഗ്രാം വഴിയായിരുന്നു ഇവർ തമ്മിൽ ബന്ധപ്പെട്ടത്. പിന്നീട് ഒരു സമ്മാനം അയക്കുന്നുണ്ടെന്നും അതിന്‍റെ ചാർജ് കൊടുക്കണമെന്നും ആവശ്യപ്പെട്ടു.സമ്മാനപ്പൊതിയില്‍ ലക്ഷങ്ങൾ ഉണ്ടെന്നും അത് കസ്‌റ്റംസ്‌ പിടിച്ചാൽ വലിയ നികുതി അടക്കേണ്ടിവരുമെന്നും യുവതിയെ ധരിപ്പിച്ചു. ഇതിനായി ഒന്നര ലക്ഷം അയക്കാൻ പറഞ്ഞു. പിന്നീട് പലതവണ ഇയാള്‍ പണം കൈക്കലാക്കി. അപകടം മനസിലാക്കി പണം അയക്കില്ലെന്ന് യുവതി പറഞ്ഞെങ്കിലും അഞ്ചുലക്ഷം രൂപ തരണമെന്ന് ആവശ്യപ്പെട്ട് ഭീഷണിപ്പെടുത്തുകയായിരുന്നു. രക്ഷയില്ലാതായതോടെ ആ പണം അടക്കം ഏഴുലക്ഷം രൂപ അക്കൗണ്ട് വഴി അയച്ചുനല്‍കിയതായി യുവതി പറയുന്നു."
print(f"Text: {text}")
print("+++++++")
print("Old Tokenizer:")
for tok in tokenizer.tokenize(text):
    print(tok, end=" ")
print("\n")
print("+++++++")
print("New Tokenizer:")
for tok in new_tokenizer.tokenize(text):
    print(tok, end=" ")
print()
