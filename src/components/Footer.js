import * as React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faLinkedin, faGithub, faKaggle, faXTwitter } from "@fortawesome/free-brands-svg-icons";

const Footer = () => {
	return (
		<nav className="navbar" role="navigation" aria-label="tertiary">
			<header className="navbar-menu">
				<section className="navbar-item" id="footer">
					<a href="https://www.linkedin.com/in/zacmessinger" target="_blank" rel="noreferrer">
						{/* <FontAwesomeIcon size="lg" icon={["fab", "linkedin"]} /> */}
						<FontAwesomeIcon size="lg" icon={faLinkedin} />
					</a>
					<a href="https://github.com/zmess24" target="_blank" rel="noreferrer">
						{/* <FontAwesomeIcon size="lg" icon={["fab", "github"]} /> */}
						<FontAwesomeIcon size="lg" icon={faGithub} />
					</a>
					<a href="https://twitter.com/zdmessinger" target="_blank" rel="noreferrer">
						{/* <FontAwesomeIcon size="lg" icon={["fab", "kaggle"]} /> */}
						<FontAwesomeIcon size="lg" icon={faXTwitter} />
					</a>
					<a href="https://www.kaggle.com/zacharymessinger" target="_blank" rel="noreferrer">
						{/* <FontAwesomeIcon size="lg" icon={["fab", "kaggle"]} /> */}
						<FontAwesomeIcon size="lg" icon={faKaggle} />
					</a>
				</section>
			</header>
		</nav>
	);
};

export default Footer;
