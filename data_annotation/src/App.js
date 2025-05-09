import React, { useEffect } from 'react';
import DataAnnotation from './DataAnnotation';
import Practice from './Practice';
import Reference from './Reference';
import Judgement from './Judgement';
import Info from './Info';
import { Routes, Route, Navigate } from 'react-router';
import Eval from './Eval';
import './App.css';
import * as THREE from 'three';
import { ErrorBoundary } from 'react-error-boundary';

function Done() {
  return (
    <div className="done-body">
      <h2>Thank you for your submission!</h2>
      <p>You can close this tab now.</p>
      <p>If you are a prolific participant, you should not have been redirected to this page. Please contact the researchers with your prolific ID.</p>
    </div>
  );
}

function NotFound() {
  return (
    <div className="not-found-body">
      <h2>404 Not Found</h2>
    </div>
  );
}

const ErrorDetails = ({error, resetErrorBoundary}) => {
  return (
    <div className="error-details">
      <h2>Oops! Something went wrong. 😔</h2>
      <div className="error-boundary-message">Please contact the researchers with your prolific ID, this URL, and the error message below.</div>
      <h3>Error Details</h3>
      <blockquote>{error.message}</blockquote>
    </div>
  );
};

function App() {
  // set the default up vector for three.js
  useEffect(() => {
    THREE.Object3D.DEFAULT_UP = new THREE.Vector3(0, 0, 1);
  }, []);

  return (
    <div className="App">
      <header className="App-header ai2-header">
        <h1>Grasp Description Annotation</h1>
      </header>
      <main className="App-main">
        <ErrorBoundary fallbackRender={ErrorDetails}>
          <Routes>
            <Route path="/" element={<DataAnnotation />} />
            <Route path="/practice" element={<Practice />} />
            <Route path="/done" element={<Done />} />
            <Route path="/reference" element={<Reference />} />
            <Route path="/judgement" element={<Judgement />} />
            <Route path="/info" element={<Info />} />
            <Route path="/404" element={<NotFound />} />
            {process.env.NODE_ENV === "development" && <Route path="/eval" element={<Eval />} />}
            <Route path="*" element={<Navigate to="/404" />} />
          </Routes>
        </ErrorBoundary>
      </main>
    </div>
  );
}

export default App;
