import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# --- Load Environment Variables ---
load_dotenv()

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Personalized Pet Care Chat", page_icon="🐾")

st.title("🐾 Personalized Pet Care Chat")
st.markdown("I'm here to help! Please tell me about your pet, then I'll give tailored recommendations!")

# --- API Key & Model Initialization ---
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("The GOOGLE_API_KEY environment variable is not set. Please check your .env file.")
        st.stop()
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7, google_api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize the LLM: {e}")
    st.stop()

# --- LangChain Components ---
template = """
You are a friendly and professional pet care expert. 
Use the information the user gave about their pet to provide tailored recommendations.

Pet Info:
{pet_info}

Conversation so far:
{chat_history}

User's request: {human_input}

Give a comprehensive bulleted list of recommendations.
"""

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="human_input",
    human_prefix="Human",
    ai_prefix="AI"
)

prompt = PromptTemplate(
    input_variables=["pet_info", "chat_history", "human_input"],
    template=template
)

conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

# --- Streamlit Session State ---
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "pet_info" not in st.session_state:
    st.session_state.pet_info = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Form UI ---
if not st.session_state.form_submitted:
    with st.form(key="pet_info_form"):
        st.subheader("Tell me about your pet")
        name = st.text_input("What's their name?")

        # ✅ Species dropdown
        species_options = ["Please select...", "Dog", "Cat", "Bird", "Rabbit", "Other"]
        species = st.selectbox("What species are they?", species_options)

        breed = ""
        other_breed = ""

        # ✅ Dynamic breed dropdowns
        if species == "Dog":
            dog_breeds = [
                "Please select...",
                "Labrador Retriever", "German Shepherd", "Golden Retriever",
                "Bulldog", "Beagle", "Poodle", "Rottweiler",
                "Dachshund", "Siberian Husky", "Other"
            ]
            selected_breed = st.selectbox("Select Dog Breed", dog_breeds)
            if selected_breed == "Other":
                other_breed = st.text_input("Please type the breed")
            breed = other_breed if selected_breed == "Other" else selected_breed

        elif species == "Cat":
            cat_breeds = [
                "Please select...",
                "Persian Cat", "Maine Coon", "Siamese Cat",
                "British Shorthair", "Bengal Cat", "Sphynx", "Ragdoll", "Other"
            ]
            selected_breed = st.selectbox("Select Cat Breed", cat_breeds)
            if selected_breed == "Other":
                other_breed = st.text_input("Please type the breed")
            breed = other_breed if selected_breed == "Other" else selected_breed

        elif species in ["Bird", "Rabbit", "Other"]:
            breed = st.text_input("Please type the breed/species")

        age = st.text_input("How old are they?")
        behavior = st.text_area("Behavior/Health concerns?")
        diet = st.text_area("Current diet?")
        exercise = st.text_area("Current exercise routine?")

        submit = st.form_submit_button("Save & Continue")

    if submit:
        if (species == "Please select...") or (not breed or breed == "Please select...") or (not age):
            st.warning("Please provide species, breed, and age.")
        else:
            st.session_state.form_submitted = True
            st.session_state.pet_info = (
                f"Name: {name}\nSpecies: {species}\nBreed: {breed}\nAge: {age}\n"
                f"Behavior/Concerns: {behavior}\nDiet: {diet}\nExercise: {exercise}"
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Thanks! What recommendations do you want — diet, exercise, or wellness tips?"
            })
            st.rerun()

# --- Chat Interface ---
else:
    # Show stored pet info as a summary card
    with st.expander("Pet Information (saved from form)", expanded=True):
        st.markdown(f"```\n{st.session_state.pet_info}\n```")

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Handle new user input
    if user_input := st.chat_input("Type your request..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Thinking..."):
            result = conversation_chain.invoke({
                "pet_info": st.session_state.pet_info,
                "human_input": user_input
            })
            response = result["text"]

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

