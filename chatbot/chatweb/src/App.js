import React, { Component } from 'react';



class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      messages: [],
      inputText: '',
    };
  }

  handleChange = (e) => {
    this.setState({ inputText: e.target.value });
  };

  addUserMessage = () => {

    var resp;
    const { inputText } = this.state;
    console.log(inputText);
    fetch('/ask', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ "input": inputText })
    })
      .then(response => response.json())
      .then(data => {
          resp=data.text;
        
      })
      .catch(error => {
        console.error('Error:', error);
      });


    
    if (inputText) {
      this.setState((prevState) => ({
        messages: [...prevState.messages, { text: inputText, type: 'user' }],
        inputText: '',
      }));
      // Simulate a bot reply (you can replace this with actual bot logic)
      setTimeout(() => {
        this.setState((prevState) => ({
          messages: [...prevState.messages, { text: resp, type: 'bot' }],
        }));
      }, 1000);
    }
  };

  render() {
    return (
      <div
        style={{
          fontFamily: 'Arial, sans-serif',
          backgroundColor: '#f4f4f4',
          margin: '0',
          padding: '0',
          display: 'flex',
          flexDirection: 'column',
          height: '100vh',
        }}
      >
        <div
          style={{
            flexGrow: 1,
            overflowY: 'auto',
            padding: '20px',
          }}
        >
          {this.state.messages.map((message, index) => (
            <div
              key={index}
              style={message.type === 'user' ? userMessageStyle : botMessageStyle}
            >
              {message.text}
            </div>
          ))}
        </div>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            padding: '10px',
            backgroundColor: '#fff',
          }}
        >
          <input
            type="text"
            placeholder="Type your message..."
            value={this.state.inputText}
            onChange={this.handleChange}
            style={{
              flexGrow: 1,
              padding: '10px',
              border: 'none',
              borderRadius: '5px',
            }}
          />
          <button
            onClick={this.addUserMessage}
            style={{
              marginLeft: '10px',
              padding: '10px',
              backgroundColor: '#0077b6',
              color: '#fff',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
            }}
          >
            Send
          </button>
        </div>
      </div>
    );
  }
}

const messageStyle = {
  padding: '10px',
  margin: '5px',
  borderRadius: '10px',
  maxWidth: '70%',
};

const userMessageStyle = {
  ...messageStyle,
  alignSelf: 'flex-start',
  backgroundColor: '#e0e0e0',
};

const botMessageStyle = {
  ...messageStyle,
  alignSelf: 'flex-end',
  backgroundColor: '#0077b6',
  color: '#fff',
};
export default App;
