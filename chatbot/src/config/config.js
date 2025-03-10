import { createChatBotMessage } from 'react-chatbot-kit';

const config = {
  initialMessages: [createChatBotMessage("Hola soy Bridgy tu asistente virtual de The Bridge ¿En qué puedo ayudarte?")],
  botName: "Chatbot",
  customStyles: {
    botMessageBox: {
      backgroundColor: "#dc0b14",
    },
    chatButton: {
      backgroundColor: "#dc0b14",
    },
  },
  customComponents: {
    botAvatar: (props) => <div className="bot-avatar">🤖</div>,
    userAvatar: () => null,
    header: () => <div className="react-chatbot-kit-chat-header">Chatbot</div>
  },
};

export default config;