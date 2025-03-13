import { createChatBotMessage } from 'react-chatbot-kit';

class ActionProvider {
  constructor(createChatBotMessage, setStateFunc, createClientMessage) {
    this.createChatBotMessage = createChatBotMessage;
    this.setState = setStateFunc;
    this.createClientMessage = createClientMessage;
  }

  handleUserMessage(message) {
    // Llamada al backend
    fetch('https://chatbot-backend-288792351129.us-central1.run.app/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
      const botMessage = this.createChatBotMessage(data.message);
      this.updateChatbotState(botMessage);
    })
    .catch(error => {
      console.error('Error:', error);
      const errorMessage = this.createChatBotMessage("Lo siento, ha ocurrido un error.");
      this.updateChatbotState(errorMessage);
    });
  }

  handleHello() {
    const message = this.createChatBotMessage("¡Hola! ¿En qué puedo ayudarte?");
    this.updateChatbotState(message);
  }

  handleHelp() {
    const message = this.createChatBotMessage("¿Qué tipo de ayuda necesitas?");
    this.updateChatbotState(message);
  }

  handleBye() {
    const message = this.createChatBotMessage("¡Hasta luego! Que tengas un buen día.");
    this.updateChatbotState(message);
  }

  updateChatbotState(message) {
    this.setState((prevState) => ({
      ...prevState,
      messages: [...prevState.messages, message],
    }));
  }
}

export default ActionProvider; 