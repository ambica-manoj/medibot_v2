export default function Message({ message, onFollowUpClick }) {
  const isUser = message.role === "user";

  // Validate follow-up questions before rendering
  // Must be non-empty strings, at least 5 chars, ideally look like questions
  const validFollowUps = () => {
    if (!Array.isArray(message.followUps)) return [];
    
    return message.followUps
      .map(q => {
        // Coerce to string and trim
        const q_str = String(q || "").trim();
        return q_str;
      })
      .filter(q => {
        // Must be at least 5 characters
        if (q.length < 5) return false;
        // Should end with punctuation or start with a question word
        const startsWithQWord = /^(what|why|how|when|where|who|which|can|could|should|will|would|is|are|do|does)\b/i.test(q);
        const endsWithPunctuation = /[?!.]$/.test(q);
        return startsWithQWord || endsWithPunctuation;
      });
  };

  const followUps = validFollowUps();

  return (
    <div className={`message ${isUser ? "user" : "bot"}`}>
      <div className="message-content">
        <p>{message.text}</p>

        {message.correctedQuery && (
          <p className="spelling-hint">
            Did you mean: <em>{message.correctedQuery}</em>?
          </p>
        )}

        {followUps.length > 0 && (
          <div className="follow-ups">
            {followUps.map((q, i) => (
              <button 
                key={i} 
                className="follow-up-chip" 
                onClick={() => onFollowUpClick(q)}
                title={q}
              >
                {q}
              </button>
            ))}
          </div>
        )}

        {message.metrics && (
          <p className="metrics">
            {message.metrics.total_ms.toFixed(0)}ms · {message.metrics.chunks_used} chunk(s)
          </p>
        )}
      </div>
    </div>
  );
}
