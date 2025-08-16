import React from 'react'
import './Footer.css'

const Footer: React.FC = () => {
  const year = new Date().getFullYear()
  return (
    <footer className="site-footer" role="contentinfo" aria-label="Footer">
      <div className="site-footer__inner">
        <p className="site-footer__text">
          © {year} <a href="https://flyovercoder.com" className="site-footer__link" target="_blank" rel="noopener noreferrer">FlyOverCoder.com</a>. All rights reserved.
        </p>
      </div>
    </footer>
  )
}

export default Footer


