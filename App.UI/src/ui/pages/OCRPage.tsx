import React from "react";

const OCRPage: React.FC = () => {
  return (
    <div style={{ padding: 16 }}>
      <h2>OCR</h2>
      <p>
        This page will host the unified training and inference pipeline for OCR,
        including a Pipeline Inspector, augmentations, and dual validation
        metrics. During migration, the full legacy OCR experience remains
        available under the Legacy route.
      </p>
    </div>
  );
};

export default OCRPage;


