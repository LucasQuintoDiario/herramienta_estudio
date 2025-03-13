class MessageParser {
  constructor(actionProvider, state) {
    this.actionProvider = actionProvider;
    this.state = state;
  }

  parse(message) {
    const lowerCase = message.toLowerCase();

    if (lowerCase.includes("hello") || lowerCase.includes("hi") || lowerCase.includes("hola") || lowerCase.includes("buenos dias")) {
      this.actionProvider.handleHello();
    } else if (lowerCase.includes("help") || lowerCase.includes("ayuda") || lowerCase.includes("necesito ayuda")) {
      this.actionProvider.handleHelp();
    } else if (lowerCase.includes("bye") || lowerCase.includes("adios") || lowerCase.includes("hasta luego")) {
      this.actionProvider.handleBye();
    } else {
      // Para cualquier otro mensaje, usamos la API de Cohere
      this.actionProvider.handleUserMessage(message);
    }
  }
}

export default MessageParser;





