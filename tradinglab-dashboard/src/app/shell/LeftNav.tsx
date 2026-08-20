import { NavLink } from 'react-router-dom';
import { NAV_ITEMS } from '../nav';

export function LeftNav() {
  return (
    <nav className="tl-nav" aria-label="Primary">
      {NAV_ITEMS.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          className={({ isActive }) =>
            `tl-nav__item ${isActive ? 'is-active' : ''}`
          }
        >
          <span className="tl-nav__glyph" aria-hidden>
            {item.glyph}
          </span>
          <span className="tl-nav__label">{item.label}</span>
          {item.offline && (
            <span className="tl-nav__offline" title="Broker bridge not configured">
              OFFLINE
            </span>
          )}
        </NavLink>
      ))}
      <div className="tl-nav__sep">Development</div>
      <NavLink
        to="/preview"
        className={({ isActive }) => `tl-nav__item ${isActive ? 'is-active' : ''}`}
      >
        <span className="tl-nav__glyph" aria-hidden>
          ◎
        </span>
        <span className="tl-nav__label">Design System</span>
      </NavLink>
    </nav>
  );
}
