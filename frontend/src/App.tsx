import React from 'react';
import './assets/css/Index.css';
import { HashRouter, Route, Routes } from "react-router-dom";
import Init from './telas/Init';
import Algoritmo from './telas/Algoritmo';
import Wrapper from './Wrapper';

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<Wrapper />}>
          <Route path="/" element={<Init />} />
          <Route path="/classifier" element={<Algoritmo />} />
        </Route>

      </Routes>
    </HashRouter>
  );
};


