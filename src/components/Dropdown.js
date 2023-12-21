import Card from "./Card";
import Link from "next/link";

const Dropdown = ({ dropDownState, data, handleClick }) => {
	let classNames = dropDownState ? "dropdown show-dropdown" : "dropdown hide-dropdown";

	return (
		<main className={classNames}>
			<nav className="navbar" role="navigation" aria-label="secondary">
				<header className="navbar-menu">
					<section className="navbar-item">
						<Link className="hamburger" href="/" onClick={handleClick} title="close">
							<span></span>
							<span></span>
							<span></span>
						</Link>
					</section>
					<section className="navbar-item navbar-brand">
						<Link href="/">Zachary Messinger</Link>
					</section>
				</header>
			</nav>
			<section className="content">
				<div id="title-section">{data.name && <h2>{data.name}</h2>}</div>
				<ul>
					{data.data.map((d) => {
						return (
							<li key={d}>
								<Card data={d} />
							</li>
						);
					})}
				</ul>
			</section>
		</main>
	);
};

export default Dropdown;
