import os
import json
import sqlite3
import glob
import asyncio
import pandas as pd
from openai import OpenAI
from rank_bm25 import BM25Okapi
import chainlit as cl
from chainlit.input_widget import Select

# ==========================================
# 1. AI SETUP & INFERENCE ENGINE
# ==========================================
GROQ_API_KEY = "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"
GEMINI_API_KEY = "AQ.Ab8RN6Jbb_LrEDiyYR5WMTGT6NVbZ-K1eqr1BmwebACbxplV8g"
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY, timeout=60.0, max_retries=0)
gemini_client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=GEMINI_API_KEY, timeout=90.0, max_retries=0)

def client_for(model_name):
    return gemini_client if "gemini" in model_name else groq_client

# Same date as the evaluation, so the database dates and policy windows match.
REFERENCE_DATE = "2026-11-09"

# Paths are built from this file's location, so the app finds the notebook's output folder
# no matter which folder Chainlit is started from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.join(BASE_DIR, "aim_output")
DB_PATH = os.path.join(WORK_DIR, "apparel_store.db")
KB_DIR = os.path.join(WORK_DIR, "knowledge_base")

# RAG Setup
kb_chunks, kb_ids = [], []
for filepath in sorted(glob.glob(os.path.join(KB_DIR, "*.md"))):
    with open(filepath, encoding="utf-8") as f:
        for n, section in enumerate(f.read().split("\n## ")):
            if section.strip():
                kb_ids.append(f"{os.path.basename(filepath)[:2]}#{n}")
                kb_chunks.append(f"[{os.path.basename(filepath)}]\n" + section.strip())

if kb_chunks:
    bm25 = BM25Okapi([c.lower().split() for c in kb_chunks])
else:
    bm25 = None
    print(f"Knowledge base files not found in {KB_DIR}. Run the notebook's data and knowledge-base cells first.")

if not os.path.exists(DB_PATH):
    print(f"Database not found at {DB_PATH}. Run the notebook's database cells first.")

def top_chunk_indexes(query: str, top_k: int = 3):
    if not bm25: return []
    scores = bm25.get_scores(query.lower().split())
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

def retrieve_policy_chunks(query: str, top_k: int = 3) -> str:
    return "\n\n---\n\n".join(kb_chunks[i] for i in top_chunk_indexes(query, top_k))

def search_policy(query: str) -> dict:
    return {"policy_context": retrieve_policy_chunks(query, top_k=3)}

def lookup_order_by_id(order_id: str) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT Order_ID, Product_Name, Category, Order_Status, Order_Date, Delivery_Date, "
            "Discount_Percent, Coupon_Code, Payment_Mode, Total_Amount, Customer_Tier "
            "FROM orders WHERE Order_ID = ?", (order_id.strip(),)).fetchone()
        conn.close()
        return dict(row) if row else {"error": f"Order {order_id} not found."}
    except Exception as e:
        return {"error": str(e)}

def lookup_product_inventory(product_id: str) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT Product_ID, Product_Name, Category, Brand, Selling_Price, Discount_Percent, Stock_Quantity "
            "FROM inventory WHERE Product_ID = ?", (product_id.strip(),)).fetchone()
        conn.close()
        return dict(row) if row else {"error": f"Product {product_id} not found."}
    except Exception as e:
        return {"error": str(e)}

TOOL_FUNCS = {"lookup_order_by_id": lookup_order_by_id, "lookup_product_inventory": lookup_product_inventory, "search_policy": search_policy}

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "lookup_order_by_id", "description": "Look up one order's live status, dates, amount, discount and payment mode. Use when an Order_ID (ORD...) is mentioned. For Shipped or Processing orders, Delivery_Date is the expected delivery date.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string", "description": "Order ID, e.g. ORD0000000639"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "lookup_product_inventory", "description": "Look up a product's live stock, selling price and discount. Use when a Product_ID (PROD...) is mentioned.", "parameters": {"type": "object", "properties": {"product_id": {"type": "string", "description": "Product ID, e.g. PROD000042"}}, "required": ["product_id"]}}},
    {"type": "function", "function": {"name": "search_policy", "description": "Search the Snitch policy documents. Use after an order lookup, with the facts that decide the case.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search words, including the order facts that matter"}}, "required": ["query"]}}},
]

SYSTEM_INSTRUCTION = f"""You are the Snitch D2C Apparel Customer Service AI Assistant. Today's reference date is {REFERENCE_DATE}.

CRITICAL RULES:
1. Call lookup_order_by_id when an Order_ID (ORD...) is mentioned, and lookup_product_inventory when a Product_ID (PROD...) is mentioned.
2. MANDATORY 2-STEP PROCESS: For ANY return, refund, or cancellation request, you are STRICTLY FORBIDDEN from answering immediately after checking the database.
   - Step 1: Call lookup_order_by_id to get the order details.
   - Step 2: You MUST then call search_policy using the order facts to check if it is allowed.
3. Apply the Snitch policy strictly based on the search_policy results.
4. If the query hits a mandatory escalation trigger, begin the reply with 'ESCALATE TO HUMAN:' and give the reason.
5. If a return, cancellation, or cash refund is completely prohibited, begin the reply with 'No' and give the policy reason.
6. NEVER mention your internal tools to the customer. Do not say "I checked the database" or "According to the policy I searched." Speak naturally like a human agent.
7. Keep replies concise and polite."""

# Keep only the most recent messages so each request stays small (and the free tier lasts longer).
MAX_HISTORY = 8

def trim_history(messages):
    system, rest = messages[0], messages[1:]
    rest = rest[-MAX_HISTORY:]
    # Start on a user message so no tool result is left without the call that made it.
    while rest and rest[0]["role"] != "user":
        rest = rest[1:]
    return [system] + rest

def assistant_turn(msg):
    # Copy each tool call back exactly as received. Gemini 3 needs its thought_signature (stored in extra_content).
    tool_calls = []
    for tc in msg.tool_calls:
        item = {"id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
        extra = (tc.model_extra or {}).get("extra_content")
        if extra:
            item["extra_content"] = extra
        tool_calls.append(item)
    return {"role": "assistant", "content": msg.content or "", "tool_calls": tool_calls}

def call_model(model_name, messages):
    # Blocking network call. It runs in a worker thread (see process_chat), so the chat stays responsive.
    kwargs = dict(model=model_name, messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto")
    if "gemini" in model_name:
        # Same settings as the evaluation: Google advises the default temperature for Gemini 3,
        # and "low" thinking keeps calls quick.
        kwargs["temperature"] = 1.0
        kwargs["reasoning_effort"] = "low"
    else:
        kwargs["temperature"] = 0.0
    return client_for(model_name).chat.completions.create(**kwargs)


# ==========================================
# 2. CHAINLIT UI & CHAT LOOP
# ==========================================

@cl.on_chat_start
async def on_chat_start():
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    cl.user_session.set("messages", messages)
    cl.user_session.set("model_name", "openai/gpt-oss-120b")

    await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="AI Provider / Model",
                values=[
                    "openai/gpt-oss-120b",
                    "openai/gpt-oss-20b",
                    "gemini-3.5-flash"
                ],
                initial_index=0,
            )
        ]
    ).send()

    actions = [
        cl.Action(name="sample_1", payload={"value": "What is the status and total amount of order ORD0000000639?"}, label="📦 Order Status"),
        cl.Action(name="sample_2", payload={"value": "I want to return my Puma T-shirt from order ORD0000003698. Am I eligible?"}, label="📄 Return Rules"),
        cl.Action(name="sample_3", payload={"value": "My order ORD0000003633 shows Delivered today, but I never received the parcel! Where is it?"}, label="🚨 Escalation"),
        cl.Action(name="sample_4", payload={"value": "Do you have PROD000042 in stock and what is the selling price?"}, label="👕 Inventory Check"),
        cl.Action(name="sample_5", payload={"value": "Can I cancel my order ORD0000000639 before it ships?"}, label="❌ Cancellation")
    ]

    await cl.Message(
        content="Hi! I am the Snitch D2C AI Assistant. Click a sample question below or type your own. \n\n⚙️ *(To change the AI model, open the **Settings** menu via the slider icon).* ",
        actions=actions
    ).send()

@cl.on_settings_update
async def update_settings(settings):
    new_model = settings["Model"]
    cl.user_session.set("model_name", new_model)
    await cl.Message(content=f"⚙️ **System:** AI engine successfully switched to `{new_model}`.").send()

@cl.action_callback("sample_1")
async def on_action_sample_1(action: cl.Action):
    await cl.Message(content=action.payload["value"], author="User").send()
    await process_chat(action.payload["value"])

@cl.action_callback("sample_2")
async def on_action_sample_2(action: cl.Action):
    await cl.Message(content=action.payload["value"], author="User").send()
    await process_chat(action.payload["value"])

@cl.action_callback("sample_3")
async def on_action_sample_3(action: cl.Action):
    await cl.Message(content=action.payload["value"], author="User").send()
    await process_chat(action.payload["value"])

@cl.action_callback("sample_4")
async def on_action_sample_4(action: cl.Action):
    await cl.Message(content=action.payload["value"], author="User").send()
    await process_chat(action.payload["value"])

@cl.action_callback("sample_5")
async def on_action_sample_5(action: cl.Action):
    await cl.Message(content=action.payload["value"], author="User").send()
    await process_chat(action.payload["value"])

@cl.on_message
async def on_message(message: cl.Message):
    await process_chat(message.content)


async def process_chat(query: str):
    model_name = cl.user_session.get("model_name")
    max_steps = 6

    messages = trim_history(cl.user_session.get("messages"))

    # 1. Base retrieval from the customer's words
    idx = top_chunk_indexes(query, top_k=3)
    retrieved = "\n\n---\n\n".join(kb_chunks[i] for i in idx) if idx else ""
    sections = [kb_ids[i] for i in idx] if idx else []

    # 2. Add the user's message to the history
    messages.append({"role": "user", "content": f"RETRIEVED POLICY CONTEXT:\n{retrieved}\n\nCUSTOMER QUERY:\n{query}"})

    msg_ui = cl.Message(content="")
    await msg_ui.send()

    # 3. Tool-calling loop
    for _ in range(max_steps):
        try:
            # Runs in a worker thread so the server keeps serving other chats while the model answers.
            resp = await asyncio.to_thread(call_model, model_name, messages)
            msg = resp.choices[0].message

            if not msg.tool_calls:
                final_answer = msg.content or ""
                msg_ui.content = final_answer
                await msg_ui.update()

                messages.append({"role": "assistant", "content": final_answer})
                cl.user_session.set("messages", messages)
                return

            messages.append(assistant_turn(msg))

            for tc in msg.tool_calls:
                ui_labels = {
                    "lookup_order_by_id": "Database: Order Lookup",
                    "lookup_product_inventory": "Database: Inventory Check",
                    "search_policy": "Knowledge Base: Policy Search"
                }
                step_name = ui_labels.get(tc.function.name, tc.function.name)

                async with cl.Step(name=step_name, type="tool") as step:
                    step.input = tc.function.arguments

                    fn = TOOL_FUNCS.get(tc.function.name)
                    try:
                        args = json.loads(tc.function.arguments)
                        out = fn(**args) if fn else {"error": "unknown tool"}

                        if tc.function.name == "search_policy":
                            hit = top_chunk_indexes(args.get("query", ""), top_k=3)
                            sections += [kb_ids[i] for i in hit]

                        step.language = "json"
                        step.output = json.dumps(out, indent=2)
                    except Exception as e:
                        out = {"error": str(e)}
                        step.output = str(e)

                    messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": json.dumps(out)})

        except Exception as e:
            msg_ui.content = f"Error connecting to AI: {str(e)}"
            await msg_ui.update()
            cl.user_session.set("messages", messages)
            return

    msg_ui.content = "Error: Max tool steps exceeded."
    await msg_ui.update()
    cl.user_session.set("messages", messages)