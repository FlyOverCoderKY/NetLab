import React from "react";
import { ThemeSwitcher } from "./ThemeSwitcher";
import { useTheme } from "../context/ThemeContext";
import "./AppHeader.css";

export interface AppHeaderProps {
  title: string;
  subtitle?: string;
}

const AppHeader: React.FC<AppHeaderProps> = ({ title, subtitle }) => {
  const theme = useTheme();

  return (
    <header
      className="visualizer-header"
      role="banner"
      aria-label="Application Header"
    >
      <div className="header-content">
        <div className="header-text">
          <h1>{title}</h1>
          {subtitle && <p>{subtitle}</p>}
        </div>
        <div className="header-controls">
          <ThemeSwitcher
            appearance={theme.appearance}
            onAppearanceChange={theme.setAppearance}
          />
        </div>
      </div>
    </header>
  );
};

export default AppHeader;
