import React, { useEffect, useState } from 'react';
import './assets/css/Index.css';
import Algoritmo from './telas/algoritmo';
import Init from './telas/Init';

export default function App() {

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
    <Init />
  );
};


