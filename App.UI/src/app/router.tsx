import React from "react";
import { Routes, Route } from "react-router-dom";
import HomePage from "../ui/pages/HomePage";
import PerceptronPage from "../ui/pages/PerceptronPage";
import OCRPage from "../ui/pages/OCRPage";
import LegacyShell from "../ui/pages/LegacyShell";

export const AppRouter: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/perceptron" element={<PerceptronPage />} />
      <Route path="/ocr" element={<OCRPage />} />
      <Route path="/legacy" element={<LegacyShell />} />
    </Routes>
  );
};

export default AppRouter;


