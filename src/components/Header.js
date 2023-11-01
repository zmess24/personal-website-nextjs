import * as React from "react";
import Link from "next/link";

const Header = ({ dropDownState, handleClick }) => {
	let classNames = dropDownState ? "alternate" : "";

	return (
		<nav className="navbar" role="navigation" aria-label="primary">
			<header className="navbar-menu" id="primary-nav">
				<section className="navbar-item">
					<Link href="/" title="open" onClick={handleClick}>
						Projects
					</Link>
					<Link href="/" title="open" onClick={handleClick}>
						Blog
					</Link>
					<Link href="/">Contact</a>
				</section>
				<section className="navbar-item navbar-brand is-hidden-mobile">
					<Link href="/" className={classNames}>
						Zachary Messinger
					</Link>
				</section>
			</header>
		</nav>
	);
};

export default Header;
