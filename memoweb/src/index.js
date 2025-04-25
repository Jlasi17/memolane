// src/index.js - Should NOT wrap <App> with <BrowserRouter>
import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';

ReactDOM.render(
  <React.StrictMode>
    <App /> {/* No Router here */}
  </React.StrictMode>,
  document.getElementById('root')
);