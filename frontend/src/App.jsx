import ChatBox from "./components/ChatBox.jsx";

export default function App() {
  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="/" aria-label="Medibot home">
          <span className="brand-mark">+</span>
          <span>Medibot</span>
        </a>
        <nav className="site-nav" aria-label="Primary navigation">
          <a href="#conversation">Conversation</a>
          <a href="#reference">Reference</a>
        </nav>
        <div className="header-status"><span className="status-dot" /> Document ready</div>
      </header>

      <main className="page-content">
        <section className="intro" aria-labelledby="page-title">
          <p className="eyebrow">Private medical reference</p>
          <h1 id="page-title">Understand your health,<br /><em>one question at a time.</em></h1>
          <p className="intro-copy">Ask clear questions about the loaded reference document and get grounded answers with sources to explore.</p>
        </section>

        <section className="workspace" id="conversation" aria-label="Medical chat workspace">
          <ChatBox />
        </section>
      </main>
    </div>
  );
}
