import csv
from io import StringIO
import zipfile
from flask import make_response
import numpy as np
import torch
import PyPDF2
import docx2txt
import os

from transformers import (GPT2LMHeadModel, GPT2Tokenizer,
                          BertTokenizer, BertForMaskedLM)
from .class_register import register_api

class AbstractLanguageChecker:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def check_probabilities(self, in_text, topk=40):
        raise NotImplementedError

    def postprocess(self, token):
        raise NotImplementedError

    def top_k_logits(logits, k):
        """
        Filters logits to only the top k choices
        from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py
        """
        if k == 0:
            return logits
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1]
        return torch.where(logits < min_values,
                       torch.ones_like(logits, dtype=logits.dtype) * -1e10,
                       logits)

@register_api(name='gpt-2-small')
class LM(AbstractLanguageChecker):
    def __init__(self, model_name_or_path="gpt2"):
        super(LM, self).__init__()
        self.enc = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.to(self.device)
        self.model.eval()
        self.start_token = self.enc(self.enc.bos_token, return_tensors='pt').data['input_ids'][0]
        print("Loaded GPT-2 model!")
    
    def check_probabilities(self, in_text, topk=40):
        # Process input
        l=[0,0,0,0]
        # create in-text to change pdf or docx to test
        token_ids = self.enc(in_text, return_tensors='pt').data['input_ids'][0]
        token_ids = torch.concat([self.start_token, token_ids])
        # Forward through the model
        output = self.model(token_ids.to(self.device))
        all_logits = output.logits[:-1].detach().squeeze()
        # construct target and pred
        all_probs = torch.softmax(all_logits, dim=1)

        y = token_ids[1:]
        # Sort the predictions for each timestep
        sorted_preds = torch.argsort(all_probs, dim=1, descending=True).cpu()
        # [(pos, prob), ...]
        real_topk_pos = list(
            [int(np.where(sorted_preds[i] == y[i].item())[0][0])
             for i in range(y.shape[0])])
        
        for i in real_topk_pos:
            if i>=1000:
                l[3]+=1
            elif i<1000 and i>=100:
                l[2]+=1
            elif i<100 and i>=10:
                l[1]+=1
            elif i<10:
                l[0]+=1
    
        return l

    def gettext(self, file, fileName):
        text = ""
        if fileName.endswith('.docx'):
            text = docx2txt.process(file)
        elif fileName.endswith('.txt'):
            new_file = open(fileName)
            text = new_file.read()
        elif fileName.endswith('.pdf'):
            # pobj = open(file, 'rb')
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in range(len(reader.pages)):
                page_text = reader.pages[page]
                text+= page_text.extract_text()
        return text

    def split_text(self, project, text):
        l=[0,0,0,0]
        l1=[0,0,0,0,0,0,0,0]
        max = len(text)
        i = 0
        while i < max:
            if i+1500 < max:
                l1 = project.lm.check_probabilities(text[i:i+1500],topk = 40)
                i+=1500
            else:
                l1 = project.lm.check_probabilities(text[i:],topk = 40)
                i = max
            for j in range(len(l1)):
                l[j] += l1[j]
        

    def check_percentage(self,project,l):
        l1=['0','0','0','0','0','0','0','0']
        for i in range(0,8,2):
            print(i)
            l1[i]=str(l[int(i/2)])
            l1[i+1] = str((l[int(i/2)]/sum(l))*100)+'%'
        return l1

    def get_values(self, project, file, filename, topk =40):
        text = project.lm.gettext(file, filename)
        l1 = [filename]
        if len(text)>2500:
            l = project.lm.split_text(project, text)
            l2 = project.lm.check_percentage(project,l)
        else:
            l = project.lm.check_probabilities(text,topk)
            l2 = project.lm.check_percentage(project,l)
        for j in range(len(l2)):
                l1.append(l2[j])
        return l1        

    def extract_files(self, project, file, topk=40):
        row=[['FileName','Top10','% Top10','Top100','% Top100','Top1000','% Top1000','Above1000','% Above1000']]

        if file.filename.endswith('.docx') or file.filename.endswith('.pdf') or file.filename.endswith('.txt'):
            l1 = project.lm.get_values(project, file, file.filename, topk)
            row.append(l1)
        elif zipfile.is_zipfile(file):
            with zipfile.ZipFile(file,'r') as zip:
                zip.extractall()
            for i in zip.infolist():
                if i.filename.endswith(".pdf") or i.filename.endswith(".docx") or i.filename.endswith(".txt") and 'MACOSX' not in i.filename:
                    l1 = project.lm.get_values(project, i.filename, i.filename, topk)
                    row.append(l1)
                else:
                    print("No valid files in zip")
        else:
            print("Its not zip or pdf or docx or text file")

        si = StringIO()
        cw = csv.writer(si)
        cw.writerows(row)
        output = make_response(si.getvalue())
        output.headers["Content-Disposition"] = "attachment; filename=result.csv"
        output.headers["Content-type"] = "text/csv"
        return output
   
    def postprocess(self, token):
        with_space = False
        with_break = False
        if token.startswith('Ġ'):
            with_space = True
            token = token[1:]
            # print(token)
        elif token.startswith('â'):
            token = ' '
        elif token.startswith('Ċ'):
            token = ' '
            with_break = True

        token = '-' if token.startswith('â') else token
        token = '“' if token.startswith('ľ') else token
        token = '”' if token.startswith('Ŀ') else token
        token = "'" if token.startswith('Ļ') else token

        if with_space:
            token = '\u0120' + token
        if with_break:
            token = '\u010A' + token

        return token

def main():
    raw_text = """"""
    