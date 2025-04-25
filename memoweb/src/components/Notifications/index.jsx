// src/components/Notifications/index.jsx
import './styles.css';
import { useState } from 'react';

const Notifications = ({ items }) => {
  if (!items || items.length === 0) {
    return <div className="notifications">No notifications</div>;
  }

  return (
    <div className="notifications">
      <h3>Notifications</h3>
      <ul className="notification-list">
        {items.map((item) => (
          <li key={item.id} className={`notification-item ${item.read ? 'read' : 'unread'}`}>
            <div className="notification-content">
              <span className="notification-type">{item.type}</span>
              <p className="notification-message">{item.message}</p>
              <time className="notification-time">
                {new Date(item.timestamp).toLocaleString()}
              </time>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Notifications;