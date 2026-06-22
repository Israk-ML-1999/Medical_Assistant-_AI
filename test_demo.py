import requests
import json

url = "http://127.0.0.1:8000/clinical-report-structure/generate"

payload = {
  "live_transcript": "Doctor: কেমন আছেন আপনি?\nPatient: ভালো আছি ডাক্তার সাহেব, কিন্তু গত দুইদিন ধরে একটু জ্বর জ্বর লাগছে আর গায়ে ব্যথা।\nDoctor: বুঝতে পেরেছি। আর কোনো সমস্যা? কাশি বা গলা ব্যথা আছে?\nPatient: না, কাশি নেই তবে মাথা ব্যথা আছে।",
  "conversation_history": [
    {
      "user_query": "hello how are you?",
      "chat_respons": "Hello! I am doing well, thank you. How can I assist you with your patient consultation today?"
    },
    {
      "user_query": "hello",
      "chat_respons": "Hello! how can i help you?"
    }
  ],
  "document_texts": [
    "Patient Name: Rahim Uddin\nAge: 45\nGender: Male\nLab Test: Complete Blood Count (CBC)\nHemoglobin: 11.5 g/dL (Low)\nFasting Blood Sugar: 5.8 mmol/L (Normal)"
  ],
  "example_structure": [
    {
      "section_name": "Patient Information",
      "details": {
        "name": "",
        "age": "",
        "gender": ""
      }
    },
    {
      "section_name": "Symptoms Identified",
      "details": {
        "primary_symptoms": [],
        "denied_symptoms": []
      }
    },
    {
      "section_name": "Lab Results Summary",
      "details": {
        "abnormal_findings": [],
        "normal_findings": []
      }
    }
  ]
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print("Status Code:", response.status_code)
try:
    print("Response JSON:\n", json.dumps(response.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print("Response Text:\n", response.text)
