// Variables globales
let sessionToken = null;
let currentQuestions = null;
const API_BASE_URL = ''; // URL base de la API (vacía si está en el mismo dominio)

// Añadir la nueva pestaña al array de tabs
const tabs = ['test', 'flashcards', 'concepts', 'materials', 'performance', 'interview'];

// Función para deshabilitar botones
function disableButtons() {
    // Deshabilitar los enlaces de la barra superior (pestañas)
    const navbarLinks = document.querySelectorAll('.navbar-nav .nav-link');
    navbarLinks.forEach(link => {
        link.style.pointerEvents = 'none';  // Deshabilita la interacción con los enlaces
        link.classList.add('disabled'); // Añadir clase para cambiar apariencia si es necesario
    });

    // Deshabilitar el botón de cerrar sesión
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.disabled = true; // Deshabilita el botón de cerrar sesión
    }

    // Deshabilitar todos los botones de formulario
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.disabled = true; // Deshabilita todos los botones
    });
}

// Función para habilitar botones
function enableButtons() {
    // Habilitar los enlaces de la barra superior (pestañas)
    const navbarLinks = document.querySelectorAll('.navbar-nav .nav-link');
    navbarLinks.forEach(link => {
        link.style.pointerEvents = 'auto';  // Vuelve a habilitar la interacción con los enlaces
        link.classList.remove('disabled'); // Elimina la clase de deshabilitado si se añadió
    });

    // Habilitar el botón de cerrar sesión
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.disabled = false; // Habilita el botón de cerrar sesión
    }

    // Habilitar todos los botones de formulario
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.disabled = false; // Habilita todos los botones
    });
}



// Función para mostrar el spinner
function showSpinner() {
    const spinner = document.getElementById('loadingSpinner');
    spinner.classList.remove('d-none'); // Muestra el spinner
}

// Función para ocultar el spinner
function hideSpinner() {
    const spinner = document.getElementById('loadingSpinner');
    spinner.classList.add('d-none'); // Oculta el spinner
}


// Función para manejar el inicio de sesión
document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    try {
        showSpinner(); // Muestra el spinner
        disableButtons(); // Deshabilita los botones
        const response = await fetch(`${API_BASE_URL}/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                nombre_usuario: username,
                password: password
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Error en el inicio de sesión');
        }

        const data = await response.json();
        sessionToken = data.session_token;
        localStorage.setItem('sessionToken', sessionToken); // Guardar el token
        document.getElementById('loginForm').classList.add('d-none');
        document.getElementById('mainContent').classList.remove('d-none');
        showSection('test');
    } catch (error) {
        alert(error.message || 'Error al conectar con el servidor');
    } finally {
        hideSpinner(); // Oculta el spinner
        enableButtons(); // Habilita los botones
    }
});

// Verificar si hay una sesión activa al cargar la página
window.addEventListener('load', () => {
    const savedToken = localStorage.getItem('sessionToken');
    if (savedToken) {
        sessionToken = savedToken;
        document.getElementById('loginForm').classList.add('d-none');
        document.getElementById('mainContent').classList.remove('d-none');
        showSection('test');
    }
});

// Función para mostrar diferentes secciones
function showSection(sectionName) {
    document.querySelectorAll('.content-section').forEach(section => {
        section.classList.add('d-none');
    });
    document.getElementById(`${sectionName}Section`).classList.remove('d-none');
}

// Event listeners para la navegación
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        showSection(e.target.dataset.page);
    });
});

// Manejar el cierre de sesión
document.getElementById('logoutBtn').addEventListener('click', () => {
    sessionToken = null;
    localStorage.removeItem('sessionToken');
    document.getElementById('mainContent').classList.add('d-none');
    document.getElementById('loginForm').classList.remove('d-none');
    document.getElementById('testSection').classList.add('d-none');
    document.getElementById('flashcardsSection').classList.add('d-none');
    document.getElementById('conceptsSection').classList.add('d-none');
    document.getElementById('performanceSection').classList.add('d-none');
    document.getElementById('recommendationsSection').classList.add('d-none');


    document.getElementById('username').value = '';
    document.getElementById('password').value = '';
});

// Generar preguntas del test
document.getElementById('testForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const topic = document.getElementById('topic').value;
    const numQuestions = document.getElementById('numQuestions').value;

    try {
        showSpinner(); // Muestra el spinner
        disableButtons(); // Deshabilita los botones
        const response = await fetch(`${API_BASE_URL}/generate-test-questions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${sessionToken}`
            },
            body: JSON.stringify({
                topic: topic,
                num_questions: parseInt(numQuestions)
            })
        });

        const data = await response.json();

        if (response.ok) {
            currentQuestions = data.questions;
            displayQuestions(data.questions);
            document.getElementById('questionsContainer').classList.remove('d-none');
            // Ocultar el feedback si se están generando nuevas preguntas
            document.getElementById('feedbackContainer').classList.add('d-none');
        } else {
            alert('Error al generar preguntas: ' + data.detail);
        }
    } catch (error) {
        alert('Error al conectar con el servidor');
    }finally {
        hideSpinner(); // Oculta el spinner
        enableButtons(); // Habilita los botones
    }
});

function displayQuestions(questions) {
    const questionsTextArea = document.getElementById('questionsText');
    let questionsText = '';

    // Si questions es un string, mostrarlo directamente
    if (typeof questions === 'string') {
        questionsText = questions;
    } else if (Array.isArray(questions)) {
        // Si es un array, procesarlo según su estructura
        questions.forEach((question) => {
            if (typeof question === 'string') {
                // Eliminar la numeración y el formato Markdown **...
                question = question.replace(/^\d+\.\s*\*\*(.*?)\*\*/g, '$1'); // Eliminar los ** (negrita) y la numeración
                if (question.trim()) {
                    // Separar el título del cuerpo de la pregunta
                    const [title, ...body] = question.split('\n');
                    questionsText += `<h3>${title}</h3>`; // Solo el título
                    if (body.length > 0) {
                        questionsText += `<p>${body.join('<br>')}</p>`; // Cuerpo de la pregunta
                    }
                }
            } else if (typeof question === 'object') {
                // Si es un objeto, intentar acceder a la propiedad que contiene el texto
                const questionText = question.text || question.question || question.content || JSON.stringify(question);
                const cleanText = questionText.replace(/^\d+\.\s*\*\*(.*?)\*\*/g, '$1'); // Eliminar los ** (negrita) y la numeración
                if (cleanText.trim()) {
                    // Separar el título del cuerpo de la pregunta
                    const [title, ...body] = cleanText.split('\n');
                    questionsText += `<h3>${title}</h3>`; // Solo el título
                    if (body.length > 0) {
                        questionsText += `<p>${body.join('<br>')}</p>`; // Cuerpo de la pregunta
                    }
                }
            }
        });
    } else if (typeof questions === 'object') {
        // Si es un objeto, convertirlo a string formateado
        questionsText = JSON.stringify(questions, null, 2);
    }

    // Asignar el texto con formato al área de texto (usamos innerHTML para HTML y no value)
    questionsTextArea.innerHTML = questionsText;

    // Limpiar el área de respuestas si había respuestas anteriores
    document.getElementById('answersText').value = '';

    // Para debug
    console.log('Preguntas recibidas:', questions);
}



// Enviar respuestas
document.getElementById('answersForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const answers = document.getElementById('answersText').value;
    
    try {
        showSpinner(); // Muestra el spinner
        disableButtons(); // Deshabilita los botones
        const response = await fetch(`${API_BASE_URL}/evaluate-answers`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${sessionToken}`
            },
            body: JSON.stringify({
                student_answers: answers,
                questions: currentQuestions
            })
        });

        const data = await response.json();
        if (response.ok) {
            displayFeedback(data.feedback);
            document.getElementById('feedbackContainer').classList.remove('d-none');
        } else {
            alert('Error al evaluar respuestas: ' + data.detail);
        }
    } catch (error) {
        alert('Error al conectar con el servidor');
    }finally {
        hideSpinner(); // Oculta el spinner
        enableButtons(); // Habilita los botones
    }
});

// Mostrar feedback con formato mejorado
function displayFeedback(feedback) {
    const container = document.getElementById('feedbackContent');
    container.innerHTML = '';
    
    if (typeof feedback === 'string') {
        // Convertir el markdown a HTML
        const formattedFeedback = feedback
            .replace(/### (.*)/g, '<h3 class="mt-4">$1</h3>')
            .replace(/#### (.*)/g, '<h4 class="mt-3">$1</h4>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/- (.*)/g, '<li>$1</li>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/---/g, '<hr>');

        container.innerHTML = `
            <div class="feedback-content">
                ${formattedFeedback}
            </div>
        `;

        // Envolver las listas en <ul>
        const lists = container.innerHTML.match(/<li>.*?<\/li>/g);
        if (lists) {
            container.innerHTML = container.innerHTML.replace(/<li>.*?<\/li>/g, match => `<ul class="mb-3">${match}</ul>`);
        }
    } else {
        // Si el feedback es un objeto con múltiples evaluaciones
        Object.entries(feedback).forEach(([question, evaluation]) => {
            const feedbackDiv = document.createElement('div');
            feedbackDiv.className = 'feedback-item mb-4';
            
            if (typeof evaluation === 'string') {
                feedbackDiv.innerHTML = `
                    <div class="card">
                        <div class="card-body">
                            <p class="card-text">${evaluation}</p>
                        </div>
                    </div>
                `;
            } else {
                feedbackDiv.innerHTML = `
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Pregunta: ${question}</h5>
                            ${evaluation.score ? `<p class="card-text"><strong>Puntuación:</strong> ${evaluation.score}</p>` : ''}
                            <p class="card-text"><strong>Feedback:</strong> ${evaluation.feedback || evaluation}</p>
                        </div>
                    </div>
                `;
            }
            
            container.appendChild(feedbackDiv);
        });
    }
}

// Generar flashcards
document.getElementById('flashcardForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const topic = document.getElementById('flashcardTopic').value;
    const numFlashcards = document.getElementById('numFlashcards').value;

    try {
        showSpinner(); // Muestra el spinner
        disableButtons(); // Deshabilita los botones
        const response = await fetch(`${API_BASE_URL}/create-flashcards`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${sessionToken}`
            },
            body: JSON.stringify({
                topic: topic,
                num_flashcards: parseInt(numFlashcards)
            })
        });

        const data = await response.json();
        console.log('Respuesta completa del backend:', data);

        if (response.ok && data.flashcards) {
            console.log('Tipo de data.flashcards:', typeof data.flashcards);
            console.log('Contenido de data.flashcards:', data.flashcards);
            displayFlashcards(data);
        } else {
            alert('Error al generar flashcards: ' + (data.detail || 'Respuesta no válida'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error al conectar con el servidor');
    }finally {
        hideSpinner(); // Oculta el spinner
        enableButtons(); // Habilita los botones
    }
});

function cleanText(text) {
    return text.replace(/\r?\n|\r/g, ' ').replace(/\s+/g, ' ').trim(); // Elimina saltos de línea y múltiples espacios
}

function parseFlashcards(markdown) {
    const flashcards = [];
    
    // Limpieza básica del texto
    markdown = cleanText(markdown);

    // Verifica si la cadena contiene el formato esperado
    console.log("Texto limpio:", markdown);
    
    // Dividimos por cada flashcard usando el delimitador "---"
    const parts = markdown.split(/---/).filter(part => part.trim() !== "");

    console.log("Partes encontradas:", parts);  // Verifica las partes divididas

    // Expresión regular ajustada para manejar los Flashcards
    const regex = /\*\*Flashcard (\d+)\*\*\s*\*Pregunta\*:\s*(.*?)\s*\*Respuesta\*:\s*(.*?)(?=\n|$)/s;

    // Procesamos cada parte
    parts.forEach((part, index) => {
        console.log(`Procesando flashcard ${index + 1}:`, part);  // Verificación de la parte que se está procesando
        const match = part.match(regex);

        if (match) {
            flashcards.push({
                flashcard: match[1].trim(),
                pregunta: match[2].trim(),
                respuesta: match[3].trim()
            });
        } else {
            console.log("No se pudo parsear:", part);  // Verifica si alguna parte no se pudo parsear
        }
    });

    console.log("Flashcards parseadas:", flashcards); // Verificación final de las flashcards
    return flashcards;
} 


function displayFlashcards(data) {
    const container = document.getElementById('flashcardsContainer');
    console.log('Contenedor de flashcards:', container);
    container.innerHTML = '';

    // Verificar si data.flashcards es un array
    if (Array.isArray(data.flashcards) && data.flashcards.length > 0) {
        data.flashcards.forEach(flashcard => {
            const cardDiv = document.createElement('div');
            cardDiv.className = 'flashcard';
            cardDiv.innerHTML = `
                <div class="flashcard-inner">
                    <div class="flashcard-front">
                        <h5>${flashcard.pregunta}</h5>
                    </div>
                    <div class="flashcard-back">
                        <p>${flashcard.respuesta}</p>
                    </div>
                </div>
            `;
            container.appendChild(cardDiv);

            // Agregar evento para voltear la flashcard al hacer clic
            cardDiv.addEventListener('click', () => {
                cardDiv.classList.toggle('flipped');
            });
            console.log('Flashcard agregada:', flashcard);
        });
    } else {
        container.innerHTML = '<p class="text-white">No se pudieron generar las flashcards o el formato es inválido.</p>';
    }

    container.classList.remove('d-none');
    console.log('Contenedor de flashcards mostrado');
}


function convertMarkdownToHTML(text) {
    return text
        // Reemplazar encabezados de nivel 3
        .replace(/### (.*)/g, '<h3 class="mt-4">$1</h3>')
        // Reemplazar encabezados de nivel 4
        .replace(/#### (.*)/g, '<h4 class="mt-3">$1</h4>')
        // Reemplazar negritas (asteriscos dobles)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Reemplazar listas con guion
        .replace(/- (.*)/g, '<li>$1</li>')
        // Reemplazar saltos de párrafo (dos saltos de línea seguidos)
        .replace(/\n\n/g, '</p><p>')
        // Reemplazar líneas horizontales
        .replace(/---/g, '<hr>')
        // Reemplazar saltos de línea simples por <br> para respetar formato
        .replace(/\n/g, '<br>');
}


document.getElementById('conceptForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const concept = document.getElementById('concept').value;

    // Mostramos la caja vacía para que se cargue con la respuesta, pero solo después de que se obtenga la explicación
    const conceptExplanation = document.getElementById('conceptExplanation');
    
    // Inicialmente ocultamos la caja de explicación
    conceptExplanation.classList.add('d-none');
    try {
        showSpinner(); // Muestra el spinner
        disableButtons(); // Deshabilita los botones
        const response = await fetch('/explain-concept', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${sessionToken}`
            },
            body: JSON.stringify({
                concept: concept
            })
        });

        const data = await response.json();
        if (response.ok) {
            let explanation = data.explanation;

            // Convertir el Markdown a HTML
            explanation = convertMarkdownToHTML(explanation);

            // Estructura la explicación
            let formattedExplanation = `
                <div class="card custom-card">
                    <div class="card-body">
                        <h5 class="card-title text-white">${concept}</h5>
                        <p class="card-text text-white">${explanation}</p>
                    </div>
                </div>
            `;

            // Si el contenido tiene un ejemplo práctico, lo estructuramos
            if (data.example) {
                formattedExplanation += `
                    <div class="mt-4">
                        <h6 class="text-white">Ejemplo Práctico:</h6>
                        <ol class="text-white">
                            ${data.example.steps.map(step => `<li>${step}</li>`).join('')}
                        </ol>
                    </div>
                `;
            }

            // Insertamos la explicación formateada
            conceptExplanation.innerHTML = formattedExplanation;

            // Mostrar la caja de la explicación solo después de que se ha cargado el contenido
            conceptExplanation.classList.remove('d-none');
        } else {
            alert('Error al explicar concepto: ' + data.detail);
        }
    } catch (error) {
        alert('Error al conectar con el servidor');
    }finally {
        hideSpinner(); // Oculta el spinner
        enableButtons(); // Habilita los botones
    }
});






// Obtener recomendaciones
document.getElementById('recommendationsForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const topic = document.getElementById('recommendationTopic').value;
    try {
        showSpinner(); // Muestra el spinner
        disableButtons(); // Deshabilita los botones
        const response = await fetch(`${API_BASE_URL}/recommend-materials`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${sessionToken}`
            },
            body: JSON.stringify({
                topic: topic,
                num_materials: 5
            })
        });

        const data = await response.json();
        if (response.ok) {
            displayRecommendations(data.recommended_materials);
        } else {
            alert('Error al obtener recomendaciones: ' + data.detail);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error al conectar con el servidor');
    }finally {
        hideSpinner(); // Oculta el spinner
        enableButtons(); // Habilita los botones
    }
});

// Mostrar recomendaciones
function displayRecommendations(recommendations) {
    const container = document.getElementById('recommendationsContainer');
    container.innerHTML = '';

    // Mostrar el contenedor solo cuando se cargan las recomendaciones
    container.classList.remove('d-none');

    if (typeof recommendations === 'string') {
        // Si es un string, formatearlo como markdown
        const formattedRecommendations = recommendations
            .replace(/### (.*)/g, '<h3 class="mt-4">$1</h3>')
            .replace(/#### (.*)/g, '<h4 class="mt-3">$1</h4>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/- (.*)/g, '<li>$1</li>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/---/g, '<hr>');

        container.innerHTML = `
            <div style="background-color: rgba(0, 0, 0, 0.9); color: white; padding: 15px; border-radius: 5px;">
                <div class="card-body">
                    ${formattedRecommendations}
                </div>
            </div>
        `;
    } else {
        recommendations.forEach(recommendation => {
            const recDiv = document.createElement('div');
            recDiv.className = 'recommendation-item mb-3';
            recDiv.innerHTML = `
                <div class="card" style="background-color: #333; color: white;">
                    <div class="card-body">
                        <h5 class="card-title">${recommendation.title || 'Recomendación'}</h5>
                        <p class="card-text">${recommendation.description || recommendation}</p>
                        ${recommendation.url ? `<a href="${recommendation.url}" target="_blank" class="btn btn-primary btn-sm">Ver recurso</a>` : ''}
                    </div>
                </div>
            `;
            container.appendChild(recDiv);
        });
    }
}


// Función para cargar el análisis de rendimiento
async function loadPerformanceAnalysis() {
    try {
        showSpinner(); // Muestra el spinner
        disableButtons(); // Deshabilita los botones
        const response = await fetch('/analyze-performance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${sessionToken}`
            },
            body: JSON.stringify({
                student_id: sessionToken
            })
        });

        const data = await response.json();
        if (response.ok) {
            // Convertir el contenido Markdown a HTML
            let formattedAnalysis = convertMarkdownToHTML(data.recommendations);

            // Mostrar el análisis de rendimiento en la interfaz
            document.getElementById('performanceAnalysis').innerHTML = `
                <div class="card custom-card">
                    <div class="card-body" style="background-color: #000; color: white; border: 1px solid #fff; border-radius: 10px;">
                        <div class="card-text text-white">${formattedAnalysis}</div>
                    </div>
                </div>
            `;
        } else {
            alert('Error al cargar análisis: ' + data.detail);
        }
    } catch (error) {
        alert('Error al conectar con el servidor');
    }finally {
        hideSpinner(); // Oculta el spinner
        enableButtons(); // Habilita los botones
    }
}

// Añadir el event listener solo al hacer clic en el botón
document.getElementById('generatePerformanceAnalysis').addEventListener('click', () => {
    loadPerformanceAnalysis();
});

let currentConversationId = null;
let isFirstInteraction = true; // Para controlar la primera interacción


// Función para formatear el feedback
function formatFeedback(feedback) {
    return feedback
        .replace(/### (.*)/g, '<h3 class="mt-4">$1</h3>')
        .replace(/#### (.*)/g, '<h4 class="mt-3">$1</h4>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/- (.*)/g, '<li>$1</li>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/---/g, '<hr>');
}

function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message) return;

    // Mostrar mensaje del usuario
    appendMessage('user', message);
    input.value = '';

    // Manejar la primera interacción
    if (isFirstInteraction) {
        const affirmativePattern = /^(sí|si|claro|por supuesto|vale|listo|ok|de acuerdo|seguro)[!.]?$/i;
        
        if (!affirmativePattern.test(message)) {
            appendMessage('assistant', "¿Listo para comenzar la entrevista?");
            return;
        }

        isFirstInteraction = false; // Desactivamos la detección de primera interacción
    }

    // Enviar mensaje al backend
    fetch('/technical-interview', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${sessionToken}`
        },
        body: JSON.stringify({
            message: message,
            conversation_id: currentConversationId,
            student_id: sessionToken
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }

        currentConversationId = data.conversation_id;

        // Si se recibió algún mensaje del entrevistador, lo mostramos
        if (data.messages && data.messages.length > 0) {
            const assistantMessage = data.messages[0].content;

            // Mostrar mensaje de la entrevista si no es el de finalización
            if (assistantMessage !== "La entrevista ha concluido. Estamos generando el feedback...") {
                appendMessage('assistant', assistantMessage);
            }
        }

        // Si la entrevista ha concluido, mostramos el mensaje de fin
        if (data.messages && data.messages[0].content === "La entrevista ha concluido. Estamos generando el feedback...") {
            appendMessage('assistant', "La entrevista ha concluido. Estamos generando el feedback...");
        }

        // Si ya tenemos el feedback, lo mostramos
        if (data.feedback) {
            const formattedFeedback = formatFeedback(data.feedback);  // Llamamos a la función para formatear el feedback
            showFeedback(formattedFeedback);  // Mostrar el feedback formateado
        }
    })
    .catch(error => {
        console.error('Error:', error);
        appendMessage('assistant', 'Lo siento, ha ocurrido un error. Por favor, intenta de nuevo.');
    });
}




document.addEventListener('DOMContentLoaded', function () {
    const initialMessage = "¡Hola! Soy tu entrevistador virtual. ¿Listo para comenzar la entrevista?";
    appendMessage('assistant', initialMessage);
});


function appendMessage(role, content) {
    const messagesContainer = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    messageDiv.textContent = content;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Función para mostrar el feedback en la interfaz
function showFeedback(formattedFeedback) {
    const feedbackContainer = document.getElementById('feedback-container');
    const feedbackContent = document.getElementById('feedback-content');
    
    feedbackContent.innerHTML = formattedFeedback;  // Usar innerHTML para insertar el HTML formateado
    feedbackContainer.style.display = 'block';
}

// Permitir enviar mensaje con Enter
document.getElementById('chat-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

