import React, { Component } from 'react';
import Chatbot from 'react-chatbot-kit';
import { ThemeProvider } from 'styled-components';
import config from '../../config/config';
import MessageParser from '../../config/MessageParser';
import ActionProvider from '../../config/ActionProvider';

const theme = {
    background: '#f5f8fb',
    header: '#dc0b14',
    botBubbleColor: '#DC3545',
    userBubbleColor: '#E9ECEF',
    botTextColor: '#ffffff',
    userTextColor: '#212529',
};


export default class Contenido extends Component {
    render() {
        return (
            <ThemeProvider theme={theme}>
                <Chatbot
                    config={config}
                    messageParser={MessageParser}
                    actionProvider={ActionProvider}
                />
            </ThemeProvider>
        )
    }
}
