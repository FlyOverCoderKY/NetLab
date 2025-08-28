import React from "react";
import "./Footer.css";

type FooterProps = {
  backend?: string;
};

const Footer: React.FC<FooterProps> = () => {
  const year = new Date().getFullYear();
  return (
    <footer className="site-footer" role="contentinfo" aria-label="Footer">
      <div className="site-footer__inner">
        <p className="site-footer__text">
          © {year}{" "}
          <a
            href="https://flyovercoder.com"
            className="site-footer__link"
            target="_blank"
            rel="noopener noreferrer"
          >
            FlyOverCoder.com
          </a>
          . All rights reserved.
        </p>
        <nav className="site-footer__links" aria-label="Footer navigation">
          <a
            href="https://www.flyovercoder.com/terms"
            className="site-footer__link"
            target="_blank"
            rel="noopener noreferrer"
          >
            Terms & Conditions
          </a>
          <span className="site-footer__separator">|</span>
          <a
            href="https://www.flyovercoder.com/privacy"
            className="site-footer__link"
            target="_blank"
            rel="noopener noreferrer"
          >
            Privacy Policy
          </a>
          <span className="site-footer__separator">|</span>
          <a
            href="https://bugs.flyovercoder.com/?p=netlab"
            className="site-footer__link"
            target="_blank"
            rel="noopener noreferrer"
          >
            Report an Issue
          </a>
        </nav>
      </div>
    </footer>
  );
};

export default Footer;
