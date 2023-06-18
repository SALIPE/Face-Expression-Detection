import React, { useEffect, useState } from 'react';
import logo from './logo.svg';
import './App.css';

function App() {

  const [emotion, setEmotion] = useState<string>("");

  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*'
  }
  const requestOptions: RequestInit = {
    method: 'GET',
    headers
  };

  useEffect(() => {
    fetch("http://localhost:5000/", requestOptions)
      .then(res => res.json())
      .then(res => setEmotion(res.emotion))
      .catch(e => console.error(e))
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.tsx</code> and save to reload.
          {emotion}
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
