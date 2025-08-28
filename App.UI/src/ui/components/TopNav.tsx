import React from "react";
import { NavLink } from "react-router-dom";

const linkStyle: React.CSSProperties = {
  padding: "8px 12px",
  textDecoration: "none",
  color: "var(--color-foreground)",
  borderRadius: 6,
};

const activeStyle: React.CSSProperties = {
  background: "var(--color-panel-background)",
  border: "1px solid var(--color-border)",
};

const containerStyle: React.CSSProperties = {
  display: "flex",
  gap: 8,
  padding: "8px 12px",
  borderBottom: "1px solid var(--color-border)",
  background: "var(--color-app-background)",
  flexWrap: "wrap",
};

const TopNav: React.FC = () => {
  return (
    <nav aria-label="Primary Navigation" style={containerStyle}>
      <NavLink
        to="/"
        style={({ isActive }) => ({
          ...linkStyle,
          ...(isActive ? activeStyle : {}),
        })}
        aria-label="Home"
      >
        Home
      </NavLink>
      <NavLink
        to="/perceptron"
        style={({ isActive }) => ({
          ...linkStyle,
          ...(isActive ? activeStyle : {}),
        })}
        aria-label="Perceptron"
      >
        Perceptron
      </NavLink>
      <NavLink
        to="/ocr"
        style={({ isActive }) => ({
          ...linkStyle,
          ...(isActive ? activeStyle : {}),
        })}
        aria-label="OCR"
      >
        OCR
      </NavLink>
      <NavLink
        to="/legacy"
        style={({ isActive }) => ({
          ...linkStyle,
          ...(isActive ? activeStyle : {}),
        })}
        aria-label="Legacy"
      >
        Legacy
      </NavLink>
    </nav>
  );
};

export default TopNav;


