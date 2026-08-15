import gradio as gr

from rag_engine import ask_rag
from document_manager import process_document


# ============================================================
# GLOBAL DOCUMENT STATE
# ============================================================

current_document = "No document uploaded"


# ============================================================
# CHAT FUNCTION
# ============================================================

def chat(message, history):

    if not message or not message.strip():
        return history, ""

    try:

        answer = ask_rag(message.strip())

        history.append({
            "role": "user",
            "content": message.strip()
        })

        history.append({
            "role": "assistant",
            "content": answer
        })

        return history, ""

    except Exception as e:

        history.append({
            "role": "user",
            "content": message.strip()
        })

        history.append({
            "role": "assistant",
            "content": f"⚠️ Error: {str(e)}"
        })

        return history, ""


# ============================================================
# CLEAR CHAT
# ============================================================

def clear_chat():
    return []


# ============================================================
# PDF UPLOAD / PROCESSING
# ============================================================

def upload_document(file):

    global current_document

    if file is None:
        return "⚠️ Please select a PDF file."

    try:

        # Gradio provides the temporary uploaded file path
        pdf_path = file

        # Make sure it is a PDF
        if not pdf_path.lower().endswith(".pdf"):
            return "⚠️ Only PDF files are supported."

        print("\n" + "=" * 60)
        print("NEW DOCUMENT UPLOAD")
        print("=" * 60)

        print(f"PDF: {pdf_path}")

        # Process document
        result = process_document(pdf_path)

        if not result["success"]:
            return f"❌ {result['message']}"

        current_document = result["filename"]

        return (
            f"✅ **Document processed successfully!**\n\n"
            f"📄 **Document:** {result['filename']}\n\n"
            f"📝 **Characters:** {result['characters']}\n\n"
            f"📚 **Chunks:** {result['chunks']}\n\n"
            f"🧠 **Embedding dimension:** {result['dimension']}\n\n"
            f"Your document is now ready. Ask a question below."
        )

    except Exception as e:

        print("UPLOAD ERROR:", str(e))

        return f"❌ Error processing document: {str(e)}"


# ============================================================
# CUSTOM CSS
# ============================================================

css = """

/* =========================
   GLOBAL
   ========================= */

body {
    margin: 0;
    background: #0f0f0f !important;
}


/* =========================
   MAIN APP
   ========================= */

.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
    background: #0f0f0f !important;
}


/* =========================
   HEADER
   ========================= */

#header {
    text-align: center;
    padding: 20px 10px 10px 10px;
}

#header h1 {
    font-size: 28px;
    margin-bottom: 5px;
}

#header p {
    color: #9ca3af;
    font-size: 14px;
}


/* =========================
   SIDEBAR
   ========================= */

#sidebar {
    background: #171717 !important;
    border: 1px solid #292929 !important;
    border-radius: 14px !important;
    padding: 18px !important;
}

.sidebar-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 12px;
}

.sidebar-text {
    color: #a3a3a3;
    font-size: 13px;
    line-height: 1.6;
}


/* =========================
   UPLOAD AREA
   ========================= */

#upload-box {
    background: #111111 !important;
    border: 1px solid #292929 !important;
    border-radius: 12px !important;
    margin-bottom: 15px !important;
}

#upload-status {
    background: #171717 !important;
    border: 1px solid #292929 !important;
    border-radius: 10px !important;
    padding: 10px !important;
}


/* =========================
   CHAT AREA
   ========================= */

#chatbot {
    border: 1px solid #292929 !important;
    border-radius: 14px !important;
    background: #111111 !important;
}


/* =========================
   INPUT AREA
   ========================= */

#input-area {
    margin-top: 12px;
}

#message-box {
    background: #1c1c1c !important;
    border: 1px solid #333333 !important;
    border-radius: 14px !important;
}

#message-box textarea {
    background: transparent !important;
    color: white !important;
    font-size: 15px !important;
}


/* =========================
   BUTTONS
   ========================= */

#send-button {
    min-width: 90px !important;
    border-radius: 12px !important;
}

#clear-button {
    width: 100%;
    border-radius: 10px !important;
}


/* =========================
   FOOTER
   ========================= */

#footer {
    text-align: center;
    color: #666666;
    font-size: 12px;
    padding: 15px;
}

"""


# ============================================================
# BUILD UI
# ============================================================

with gr.Blocks(
    title="RAG AI Assistant",
    css=css,
    theme=gr.themes.Base(
        primary_hue="slate",
        neutral_hue="slate"
    )
) as app:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.HTML(
        """
        <div id="header">
            <h1>🤖 RAG AI Assistant</h1>

            <p>
                Upload a document and ask questions about its content.
            </p>
        </div>
        """
    )


    # --------------------------------------------------------
    # MAIN LAYOUT
    # --------------------------------------------------------

    with gr.Row():

        # ====================================================
        # SIDEBAR
        # ====================================================

        with gr.Column(
            scale=1,
            min_width=220,
            elem_id="sidebar"
        ):

            gr.HTML(
                """
                <div class="sidebar-title">
                    📄 Document Assistant
                </div>

                <div class="sidebar-text">

                    <b>Retrieval</b><br>
                    FAISS Vector Search

                    <br><br>

                    <b>Embedding model</b><br>
                    all-MiniLM-L6-v2

                    <br><br>

                    <b>AI model</b><br>
                    Gemini

                    <br><br>

                    Upload a PDF and ask questions
                    based on its contents.

                </div>
                """
            )


            # ------------------------------------------------
            # PDF UPLOAD
            # ------------------------------------------------

            upload = gr.File(
                label="📎 Upload PDF",
                file_types=[".pdf"],
                type="filepath",
                elem_id="upload-box"
            )


            upload_status = gr.Markdown(
                value="📄 **No document uploaded**",
                elem_id="upload-status"
            )


            # ------------------------------------------------
            # CLEAR CHAT
            # ------------------------------------------------

            clear_button = gr.Button(
                "🗑️ Clear Chat",
                elem_id="clear-button"
            )


        # ====================================================
        # CHAT SECTION
        # ====================================================

        with gr.Column(
            scale=4
        ):

            chatbot = gr.Chatbot(
                label="Conversation",
                height=570,
                elem_id="chatbot"
            )


            # ------------------------------------------------
            # INPUT
            # ------------------------------------------------

            with gr.Row(
                elem_id="input-area"
            ):

                message = gr.Textbox(
                    placeholder="Ask something about your document...",
                    show_label=False,
                    lines=2,
                    max_lines=5,
                    scale=8,
                    elem_id="message-box"
                )

                send_button = gr.Button(
                    "➤",
                    variant="primary",
                    scale=1,
                    elem_id="send-button"
                )


    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    gr.HTML(
        """
        <div id="footer">
            RAG AI Assistant • Answers are generated from
            retrieved document context
        </div>
        """
    )


    # ========================================================
    # EVENTS
    # ========================================================

    # PDF upload
    upload.change(
        fn=upload_document,
        inputs=[upload],
        outputs=[upload_status]
    )


    # Send button
    send_button.click(
        fn=chat,
        inputs=[message, chatbot],
        outputs=[chatbot, message]
    )


    # Enter key
    message.submit(
        fn=chat,
        inputs=[message, chatbot],
        outputs=[chatbot, message]
    )


    # Clear chat
    clear_button.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot]
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    app.launch(
        inbrowser=True
    )