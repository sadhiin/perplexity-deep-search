# AI Chat Assistant Frontend

A minimalist, modern chat interface for AI conversations with chat history management.

## Features

- **Clean Minimalist Design**: Modern UI with subtle gradients and smooth animations
- **Chat History Sidebar**: Left sidebar showing all previous conversations
- **Dark Mode Support**: Toggle between light and dark themes with persistent preference
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- **Real-time Chat**: Instant message sending with typing indicators
- **Auto-resizing Input**: Textarea that grows with your message
- **Character Counter**: Shows message length (max 2000 characters)
- **Local Storage**: Saves chat history and theme preference in browser storage
- **Mobile Menu**: Collapsible sidebar for mobile devices

## File Structure

```text
frontend/
├── index.html      # Main HTML structure
├── styles.css      # Modern minimalist styling
├── script.js       # Chat functionality and state management
└── README.md       # This file
```

## Usage

1. **Open `index.html`** in your web browser
2. **Start chatting** by typing in the input field at the bottom
3. **Create new chats** using the "New Chat" button in the sidebar
4. **Browse chat history** by clicking on previous conversations in the sidebar
5. **Mobile users** can toggle the sidebar using the menu button

## Key Features

### Chat Interface

- Send messages with Enter key or send button
- Shift+Enter for new lines
- Auto-scrolling to latest messages
- Typing indicators during AI responses

### Chat Management

- Automatic chat titling based on first message
- Timestamp display for each message
- Persistent chat history using localStorage
- Easy switching between conversations

### Dark Mode

- Toggle between light and dark themes using the moon/sun icon
- Theme preference saved in localStorage
- Smooth transitions between themes
- Optimized color schemes for both modes

### Responsive Design

- Desktop: Full sidebar always visible
- Mobile: Collapsible sidebar with overlay
- Optimized touch targets for mobile devices

## Customization

### Styling

The design uses CSS custom properties for easy theming:

- Primary gradient: `--primary-gradient`
- Background colors: `--bg-primary`, `--bg-secondary`
- Text colors: `--text-primary`, `--text-secondary`

### Backend Integration

Replace the `getAIResponse()` function in `script.js` with your actual API call:

```javascript
async function getAIResponse(message) {
    const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message })
    });

    const data = await response.json();
    return data.response;
}
```

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Development

To run locally with a simple server:

```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx http-server -p 8000

# Then open http://localhost:8000/frontend/
```

## License

This frontend is part of the Perplexity Deep Search project.
