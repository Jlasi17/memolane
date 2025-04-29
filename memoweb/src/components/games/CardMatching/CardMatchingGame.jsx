import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './CardMatchingGame.css';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const CardMatchingGame = () => {
  const navigate = useNavigate();
  const [cards, setCards] = useState([]);
  const [flipped, setFlipped] = useState([]);
  const [matched, setMatched] = useState([]);
  const [moves, setMoves] = useState(0);
  const [gameOver, setGameOver] = useState(false);
  const [timer, setTimer] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [difficulty, setDifficulty] = useState('easy');
  const [score, setScore] = useState(0);
  const [highScores, setHighScores] = useState([]);
  const [playerLevel, setPlayerLevel] = useState(1);
  const [playerExp, setPlayerExp] = useState(0);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState('');
  const [showLevelUp, setShowLevelUp] = useState(false);
  const [newLevel, setNewLevel] = useState(1);

  // Game configurations based on difficulty and player level
  const difficultyConfig = {
    easy: {
      cardCount: 12 + Math.min(4, Math.floor(playerLevel / 3)), // Scales with level
      timeLimit: 180, // 3 minutes
      matchPoints: 10,
      timeBonusDivisor: 15 // Points per X seconds remaining
    },
    medium: {
      cardCount: 16 + Math.min(6, Math.floor(playerLevel / 2)),
      timeLimit: 150,
      matchPoints: 15,
      timeBonusDivisor: 12
    },
    hard: {
      cardCount: 20 + Math.min(8, playerLevel),
      timeLimit: 120,
      matchPoints: 20,
      timeBonusDivisor: 10
    },
    expert: {
      cardCount: 24 + Math.min(12, playerLevel * 2),
      timeLimit: 90,
      matchPoints: 25,
      timeBonusDivisor: 8
    }
  };

  // Symbols for cards (emoji or images)
  const symbols = [
    '🍎', '🍌', '🍒', '🍓', '🍊', '🍋', '🍍', '🥝', 
    '🍇', '🍉', '🍐', '🥥', '🍑', '🥭', '🍈', '🍏',
    '🥕', '🍆', '🥑', '🥦', '🧄', '🧅', '🥔', '🌽'
  ];

  // Initialize the game
const initializeGame = async () => {
  setLoading(true);
  try {
    const token = localStorage.getItem('token');
    if (!token) {
      throw new Error('No authentication token found');
    }

    // First get the current user's basic info to get their patient_id
    const userInfoResponse = await fetch(`${API_BASE_URL}/api/user`, {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    });
    
    if (!userInfoResponse.ok) {
      throw new Error('Failed to fetch user info');
    }
    
    const userInfo = await userInfoResponse.json();
    const patientId = userInfo.username; // Assuming username is the patient_id

    // Now fetch the game user profile using the patient_id
    const gameUserResponse = await fetch(`${API_BASE_URL}/api/game_user/${patientId}`, {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    });
    
    if (!gameUserResponse.ok) {
      // If we can't get the user profile, use defaults but still let the game start
      console.error('Failed to fetch game user data, using defaults');
      setPlayerLevel(1);
      setPlayerExp(0);
    } else {
      const userData = await gameUserResponse.json();
      setPlayerLevel(userData.level);
      setPlayerExp(userData.exp);
    }

    // Continue with game initialization
    const config = difficultyConfig[difficulty];
    const pairs = symbols.slice(0, config.cardCount / 2);
    const deck = [...pairs, ...pairs];
    
    // Shuffle the deck
    for (let i = deck.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [deck[i], deck[j]] = [deck[j], deck[i]];
    }

    setCards(deck);
    setFlipped([]);
    setMatched([]);
    setMoves(0);
    setScore(0);
    setTimer(0);
    setIsRunning(true);
    setGameOver(false);
    setMessage('');
  } catch (error) {
    console.error("Error initializing game:", error);
    setMessage("Failed to initialize game - using default settings");
    // Set default values if there's an error
    setPlayerLevel(1);
    setPlayerExp(0);
  } finally {
    setLoading(false);
  }
};

  // Handle card click
  const handleCardClick = (index) => {
    if (loading || gameOver || flipped.includes(index) || matched.includes(index) || flipped.length === 2) {
      return;
    }

    const newFlipped = [...flipped, index];
    setFlipped(newFlipped);

    if (newFlipped.length === 2) {
      setMoves(moves + 1);
      
      if (cards[newFlipped[0]] === cards[newFlipped[1]]) {
        // Match found
        const newMatched = [...matched, ...newFlipped];
        setMatched(newMatched);
        setFlipped([]);
        
        const config = difficultyConfig[difficulty];
        setScore(score + config.matchPoints);
        
        // Check for game completion
        if (newMatched.length === cards.length) {
          endGame(true);
        }
      } else {
        // No match
        setTimeout(() => setFlipped([]), 1000);
      }
    }
  };

  // End game handler
  const endGame = async (completed) => {
    setIsRunning(false);
    setGameOver(true);
    
    if (completed) {
      const config = difficultyConfig[difficulty];
      
      // Calculate time bonus
      const timeRemaining = config.timeLimit - timer;
      const timeBonus = Math.floor(timeRemaining / config.timeBonusDivisor);
      const finalScore = score + timeBonus;
      
      try {
        setLoading(true);
        const response = await fetch(`${API_BASE_URL}/api/save-memory-score`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            score: finalScore,
            level: playerLevel,
            difficulty: difficulty,
            time: timer,
            time_limit: config.timeLimit,
            matches: matched.length / 2 + 1,
            moves: moves
          })
        });

        const data = await response.json();
        
        if (data.success) {
          setMessage(data.is_high_score ? '🌟 New High Score! 🌟' : 'Game completed!');
          
          if (data.levels_gained > 0) {
            setNewLevel(data.new_level);
            setShowLevelUp(true);
          }
          
          // Update player stats
          setPlayerLevel(data.new_level);
          setPlayerExp(prev => prev + data.exp_gained);
        }
        
        // Refresh high scores
        fetchHighScores();
      } catch (error) {
        console.error('Error saving score:', error);
        setMessage('Failed to save score');
      } finally {
        setLoading(false);
      }
    } else {
      setMessage("Time's up!");
    }
  };

  // Fetch high scores
  const fetchHighScores = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/memory-high-scores?difficulty=${difficulty}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setHighScores(data.scores || []);
      }
    } catch (error) {
      console.error('Error fetching high scores:', error);
    }
  };

  // Timer effect
  useEffect(() => {
    let interval;
    
    if (isRunning && !gameOver) {
      interval = setInterval(() => {
        setTimer(prev => {
          const newTime = prev + 1;
          if (newTime >= difficultyConfig[difficulty].timeLimit) {
            endGame(false);
            return prev;
          }
          return newTime;
        });
      }, 1000);
    }
    
    return () => clearInterval(interval);
  }, [isRunning, gameOver, difficulty]);

  // Initialize game on mount and when difficulty changes
  useEffect(() => {
    initializeGame();
  }, [difficulty]);

  // Format time display
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  // Calculate progress to next level
  const expPercentage = playerLevel > 0 
    ? Math.min(100, (playerExp / (playerLevel * 100)) * 100)
    : 0;

  if (loading && cards.length === 0) {
    return (
      <div className="game-loading">
        <div className="spinner"></div>
        <p>Loading game...</p>
      </div>
    );
  }

  return (
    <div className="card-game-container">
      {/* Level Up Modal */}
      {showLevelUp && (
        <div className="level-up-modal">
          <div className="level-up-content">
            <div className="confetti">
              {[...Array(50)].map((_, i) => (
                <div key={i} className="confetti-piece" />
              ))}
            </div>
            <h2>Level Up!</h2>
            <h1 className="level-display">Level {newLevel}</h1>
            <p>Congratulations on your achievement!</p>
            <button 
              className="close-button"
              onClick={() => setShowLevelUp(false)}
            >
              Continue Playing
            </button>
          </div>
        </div>
      )}

      {/* Game Header */}
      <header className="game-header">
        <button className="back-button" onClick={() => navigate('/patient')}>
          ← Back to Home
        </button>
      </header>

      {/* Game Controls */}
      <div className="game-controls">
        <div className="difficulty-selector">
          <label>Difficulty:</label>
          <select 
            value={difficulty} 
            onChange={(e) => setDifficulty(e.target.value)}
            disabled={isRunning && !gameOver}
          >
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
            <option value="expert">Expert</option>
          </select>
        </div>
        
        <div className="game-stats">
          <div className="stat">
            <span className="stat-label">Time:</span>
            <span className="stat-value">
              {formatTime(timer)} / {formatTime(difficultyConfig[difficulty].timeLimit)}
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">Moves:</span>
            <span className="stat-value">{moves}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Score:</span>
            <span className="stat-value">{score}</span>
          </div>
        </div>
        
        <button 
          className="restart-button"
          onClick={initializeGame}
          disabled={loading}
        >
          {loading ? 'Loading...' : 'Restart Game'}
        </button>
      </div>

      {/* Game Message */}
      {message && (
        <div className="game-message">
          {message}
        </div>
      )}

      {/* Game Board */}
      <div 
        className="card-grid" 
        style={{
          gridTemplateColumns: `repeat(${Math.ceil(Math.sqrt(difficultyConfig[difficulty].cardCount))}, 1fr)`
        }}
      >
        {cards.map((symbol, index) => {
          const isFlipped = flipped.includes(index) || matched.includes(index);
          const isMatched = matched.includes(index);
          
          return (
            <div
              key={index}
              className={`card ${isFlipped ? 'flipped' : ''} ${isMatched ? 'matched' : ''}`}
              onClick={() => handleCardClick(index)}
            >
              <div className="card-inner">
                <div className="card-front">
                  {symbol}
                </div>
                <div className="card-back">
                  <div className="card-back-pattern"></div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Game Over Modal */}
      {gameOver && (
        <div className="game-overlay">
          <div className="game-over-modal">
            <h2>{matched.length === cards.length ? '🎉 You Won! 🎉' : "Game Over"}</h2>
            
            <div className="result-stats">
              <div className="result-stat">
                <span className="stat-label">Final Score:</span>
                <span className="stat-value">{score}</span>
              </div>
              <div className="result-stat">
                <span className="stat-label">Time:</span>
                <span className="stat-value">{formatTime(timer)}</span>
              </div>
              <div className="result-stat">
                <span className="stat-label">Matches:</span>
                <span className="stat-value">{matched.length / 2}</span>
              </div>
              <div className="result-stat">
                <span className="stat-label">Moves:</span>
                <span className="stat-value">{moves}</span>
              </div>
            </div>

            <div className="high-scores">
              <h3>Your High Scores ({difficulty})</h3>
              {highScores.length > 0 ? (
                <ol>
                  {highScores.map((hs, i) => (
                    <li key={i} className={hs.is_high_score ? 'highlight' : ''}>
                      <span className="score-rank">{i + 1}.</span>
                      <span className="score-value">{hs.score}</span>
                      <span className="score-details">
                        (Lv {hs.level}, {formatTime(hs.time)})
                      </span>
                    </li>
                  ))}
                </ol>
              ) : (
                <p>No high scores yet!</p>
              )}
            </div>

            <div className="modal-actions">
              <button 
                className="play-again-button"
                onClick={initializeGame}
                disabled={loading}
              >
                Play Again
              </button>
              <button 
                className="return-button"
                onClick={() => navigate('/patient')}
              >
                Return to Dashboard
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CardMatchingGame;