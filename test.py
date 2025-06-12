import streamlit as st
import boto3
import json
from botocore.config import Config


# กำหนด client พร้อมขยาย timeout
client = boto3.client(
   "bedrock-runtime",
   region_name="us-east-1",
   config=Config(connect_timeout=3600, read_timeout=3600, retries={'max_attempts': 1})
)


# เปลี่ยนมาใช้ Amazon Nova Lite
MODEL_ID = "amazon.nova-lite-v1:0"


st.title("AI สรุปรายเอกสาร (Powered by Amazon Nova Lite)")


# รับข้อความดิบจากผู้ใช้
input_text = st.text_area("กรุณาใส่เนื้อหาเอกสาร (RAW TEXT):", height=250)


if st.button("สรุปให้เลย!"):
   # กำหนด system prompt เพื่อสั่งให้สรุปเป็นภาษาไทยอย่างกระชับ
   system_prompts = [
       {
           "text": "Act as a concise summarization assistant. สรุปรายงานนี้เป็นภาษาไทยให้สั้นและชัดเจน"
       }
   ]
   # สร้างรายการ messages ตาม schema
   messages = [
       {
           "role": "user",
           "content": [{"text": input_text}]
       }
   ]
   # กำหนดพารามิเตอร์การ inference
   inference_config = {
       "maxTokens": 1000,
       "temperature": 0.5,
       "topP": 0.9,
       "topK": 40
   }


   # ประกอบ request body ตาม “messages-v1”
   request_body = {
       "schemaVersion": "messages-v1",
       "system": system_prompts,
       "messages": messages,
       "inferenceConfig": inference_config
   }


   # เรียก Bedrock API แบบ buffered (ไม่ใช้ stream)
   response = client.invoke_model(
       modelId=MODEL_ID,
       
       contentType="application/json",
       accept="application/json",
       body=json.dumps(request_body)
   )


   # แปลงผลลัพธ์เป็น dict แล้วดึงข้อความ summary
   result = json.loads(response["body"].read())


   # ดึงเฉพาะข้อความคำตอบ
   try:
       summary = result["output"]["message"]["content"][0]["text"]
   except (KeyError, IndexError):
       # กรณีโครงสร้างไม่ตรงตามคาด ให้ fallback ไปใช้แบบเดิม
       summary = result.get("results", [])[0].get("message", {}).get("content", [])[0].get("text", "")
   # แสดงผลลัพธ์
   st.subheader("ผลลัพธ์:")
   st.write(summary)
