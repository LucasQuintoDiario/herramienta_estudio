class MessageParser {
  constructor(actionProvider, state) {
    this.actionProvider = actionProvider;
    this.state = state;
  }

  parse(message) {
    const lowerCase = message.toLowerCase();

    if (lowerCase.includes("hello") || lowerCase.includes("hi")) {
      this.actionProvider.handleHello();
    } else if (lowerCase.includes("help")) {
      this.actionProvider.handleHelp();
    } else if (lowerCase.includes("bye")) {
      this.actionProvider.handleBye();
    } else {
      // Para cualquier otro mensaje, usamos la API de Cohere
      this.actionProvider.handleUserMessage(message);
    }
  }
}

export default MessageParser;