from __future__ import annotations
import argparse,csv,json,math,random,re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
WORD_ID_RE=re.compile(r"^(.*)_(\d+)_page_(\d+)_(\d+)$")
DEFAULT_MODEL="dumitrescustefan/bert-base-romanian-cased-v1"
@dataclass(frozen=True)
class WordChunk:
    words:tuple[str,...]; row_indices:tuple[int,...]; targets:tuple[float,...]|None; datapoint_ids:tuple[str,...]
def read_rows(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8-sig") as handle: return list(csv.DictReader(handle))
def parse_word_id(word_id:str)->tuple[int,int,int]:
    match=WORD_ID_RE.match(word_id); return (int(match.group(2)),int(match.group(3)),int(match.group(4))) if match else (0,0,0)
def chunk_rows(rows,labeled:bool,chunk_words:int)->list[WordChunk]:
    groups:OrderedDict[tuple[str,str,int,int],list[tuple[int,dict[str,str]]]]=OrderedDict()
    for row_index,row in enumerate(rows):
        source,page,word_index=parse_word_id(row["word_id"]); key=(row["participant_id"],row["text"],source,page)
        groups.setdefault(key,[]).append((word_index,{**row,"_row_index":str(row_index)}))
    chunks=[]
    for values in groups.values():
        ordered=sorted(values,key=lambda item:item[0])
        for start in range(0,len(ordered),chunk_words):
            part=[row for _,row in ordered[start:start+chunk_words]]
            targets=tuple(float(row["answer"]) for row in part) if labeled else None
            ids=tuple(row.get("datapointID",row["_row_index"]) for row in part)
            chunks.append(WordChunk(tuple(row["word"] for row in part),tuple(int(row["_row_index"]) for row in part),targets,ids))
    return chunks
def first_token_positions(word_ids:list[int|None],word_count:int)->list[int]:
    positions=[-1]*word_count
    for token_index,word_id in enumerate(word_ids):
        if word_id is not None and 0<=word_id<word_count and positions[word_id]<0: positions[word_id]=token_index
    if any(value<0 for value in positions): raise ValueError("tokenizer did not preserve every input word")
    return positions
def combine_predictions(read_logits:np.ndarray,log_times:np.ndarray)->np.ndarray:
    read_prob=1.0/(1.0+np.exp(-read_logits.astype(float))); positive_ms=np.expm1(np.clip(log_times.astype(float),0.0,10.0))
    return np.clip(read_prob*positive_ms,0.0,10000.0)
class BertTwoHead(torch.nn.Module):
    def __init__(self,model_name:str,dropout:float,unfreeze_layers:int):
        super().__init__(); self.encoder=AutoModel.from_pretrained(model_name); hidden=self.encoder.config.hidden_size
        self.dropout=torch.nn.Dropout(dropout); self.read_head=torch.nn.Linear(hidden,1); self.time_head=torch.nn.Linear(hidden,1)
        for param in self.encoder.parameters(): param.requires_grad=False
        layers=getattr(getattr(self.encoder,"encoder",None),"layer",[])
        for layer in list(layers)[-max(0,unfreeze_layers):]:
            for param in layer.parameters(): param.requires_grad=True
    def forward(self,input_ids,attention_mask,**unused):
        hidden=self.dropout(self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state)
        return self.read_head(hidden).squeeze(-1),torch.nn.functional.softplus(self.time_head(hidden).squeeze(-1))
def select_device(requested:str)->str:
    if requested!="auto": return requested
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"
def encode_chunk(tokenizer,chunk:WordChunk,device:str,max_length:int):
    encoded=tokenizer(list(chunk.words),is_split_into_words=True,return_tensors="pt",truncation=True,max_length=max_length)
    positions=torch.as_tensor(first_token_positions(encoded.word_ids(),len(chunk.words)),dtype=torch.long,device=device)
    tensors={key:value.to(device) for key,value in dict(encoded).items()}
    return tensors,positions
def train_model(args,chunks:list[WordChunk]):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    tokenizer=AutoTokenizer.from_pretrained(args.model); device=select_device(args.device); model=BertTwoHead(args.model,args.dropout,args.unfreeze_layers).to(device)
    random.Random(args.seed).shuffle(chunks); optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate)
    bce=torch.nn.BCEWithLogitsLoss(); mse=torch.nn.MSELoss(); model.train()
    for _ in range(args.epochs):
        for chunk in chunks:
            tensors,positions=encode_chunk(tokenizer,chunk,device,args.max_length); read_logits,log_times=model(**tensors)
            read_word=read_logits[0,positions]; time_word=log_times[0,positions]; target=torch.as_tensor(chunk.targets,dtype=torch.float32,device=device)
            read_target=(target>0.0).float(); positive=target>0.0; time_loss=mse(time_word[positive],torch.log1p(target[positive])) if bool(positive.any()) else time_word.sum()*0.0
            loss=args.skip_weight*bce(read_word,read_target)+args.time_weight*time_loss; optimizer.zero_grad(); loss.backward(); optimizer.step()
    return tokenizer,model,device
def predict_chunks(args,backend,chunks:list[WordChunk],row_count:int)->np.ndarray:
    tokenizer,model,device=backend; predictions=np.zeros(row_count,dtype=float); filled=np.zeros(row_count,dtype=bool); model.eval()
    with torch.no_grad():
        for chunk in chunks:
            tensors,positions=encode_chunk(tokenizer,chunk,device,args.max_length); read_logits,log_times=model(**tensors)
            values=combine_predictions(read_logits[0,positions].detach().cpu().numpy(),log_times[0,positions].detach().cpu().numpy())
            for row_index,value in zip(chunk.row_indices,values,strict=True): predictions[row_index]=value; filled[row_index]=True
    if not np.all(filled): raise ValueError("each test row must receive a prediction")
    return np.clip(predictions*args.prediction_scale,0.0,10000.0)
def write_submission(rows,predictions:np.ndarray,output_path:Path)->None:
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["subtaskID","datapointID","answer"],lineterminator="\n"); writer.writeheader()
        for row,prediction in zip(rows,predictions,strict=True): writer.writerow({"subtaskID":"1","datapointID":row["datapointID"],"answer":f"{max(0.0,float(prediction)):.6f}"})
def main():
    parser=argparse.ArgumentParser(description="Generate BERT two-head scale 1.20 candidate.")
    parser.add_argument("--train",type=Path,default=Path("data/train_data.csv")); parser.add_argument("--test",type=Path,default=Path("data/test_data.csv")); parser.add_argument("--output",type=Path,default=Path("submission.csv"))
    parser.add_argument("--model",default=DEFAULT_MODEL); parser.add_argument("--device",default="auto"); parser.add_argument("--epochs",type=int,default=1); parser.add_argument("--learning-rate",type=float,default=2e-5)
    parser.add_argument("--dropout",type=float,default=0.1); parser.add_argument("--unfreeze-layers",type=int,default=1); parser.add_argument("--chunk-words",type=int,default=96); parser.add_argument("--max-length",type=int,default=512)
    parser.add_argument("--skip-weight",type=float,default=0.5); parser.add_argument("--time-weight",type=float,default=1.0); parser.add_argument("--seed",type=int,default=7); parser.add_argument("--prediction-scale",type=float,default=1.2); parser.add_argument("--report",type=Path)
    args=parser.parse_args(); train_rows=read_rows(args.train); test_rows=read_rows(args.test); train_chunks=chunk_rows(train_rows,True,args.chunk_words); test_chunks=chunk_rows(test_rows,False,args.chunk_words)
    predictions=predict_chunks(args,train_model(args,train_chunks),test_chunks,len(test_rows)); write_submission(test_rows,predictions,args.output)
    if args.report: args.report.write_text(json.dumps({"mode":"bert_two_head_official_candidate","score_type":"local_estimate","train_chunks":len(train_chunks),"test_rows":len(test_rows),"prediction_scale":args.prediction_scale,"promoted":False},indent=2,sort_keys=True)+"\n",encoding="utf-8")
if __name__=="__main__": main()
