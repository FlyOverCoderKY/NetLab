import React from "react";
import { Link } from "react-router-dom";

const HomePage: React.FC = () => {
  return (
    <div style={{ padding: 16 }}>
      <h2>Neural Networks</h2>
      <p>
        Explore key concepts of neural networks through interactive demos. Start
        with a simple perceptron to understand weighted inputs and activation
        functions, then train a small model to perform optical character
        recognition (OCR).
      </p>
      <div
        style={{ display: "flex", gap: 12, marginTop: 12, flexWrap: "wrap" }}
      >
        <Link to="/perceptron" style={{ textDecoration: "none" }}>
          <button>Try Perceptron</button>
        </Link>
        <Link to="/ocr" style={{ textDecoration: "none" }}>
          <button>Go to OCR</button>
        </Link>
        <Link to="/legacy" style={{ textDecoration: "none" }}>
          <button>Open Legacy UI</button>
        </Link>
      </div>
    </div>
  );
};

export default HomePage;


